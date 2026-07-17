"""
etl/segmenter.py
────────────────
AI Inference Node — runs 3D segmentation on the normalized volumes.

Supports:
  1. Mock Engine: Clinically realistic morphological segmentation using scipy
     (ventricles for brain MRI, bone structures for CT).
  2. ONNX Engine: Loads a 3D U-Net model from an ONNX path if available.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from loguru import logger

try:
    import scipy.ndimage as ndimage
except ImportError:
    ndimage = None  # fallback handling


class DicomSegmenter:
    """
    Runs 3D AI segmentation inference on normalized image volumes.
    """

    def __init__(
        self,
        enabled: bool = True,
        model_type: str = "mock",
        model_path: str = "",
        threshold: float = 0.5,
        target_structure: str = "ventricles",
    ):
        self.enabled = enabled
        self.model_type = model_type
        self.model_path = model_path
        self.threshold = threshold
        self.target_structure = target_structure
        self.onnx_session = None

        if self.enabled and self.model_type == "onnx":
            self._init_onnx()

    def _init_onnx(self) -> None:
        """Attempt to load the ONNX runtime session."""
        try:
            import onnxruntime as ort
            if os.path.exists(self.model_path):
                logger.info(f"Loading 3D U-Net ONNX model from: {self.model_path}")
                self.onnx_session = ort.InferenceSession(self.model_path)
            else:
                logger.warning(
                    f"ONNX model file not found at '{self.model_path}'. "
                    f"Falling back to Mock Engine."
                )
        except ImportError:
            logger.warning(
                "onnxruntime not installed. Falling back to Mock Engine. "
                "To use ONNX: pip install onnxruntime"
            )

    def segment(self, volume: np.ndarray, modality: str, spacing: tuple[float, float, float]) -> tuple[np.ndarray, float]:
        """
        Segment the 3D volume.

        Args:
            volume: Normalized 3D numpy array.
            modality: "MR", "CT", etc.
            spacing: Voxel spacing (dx, dy, dz) in mm.

        Returns:
            Tuple of:
              - binary_mask: 3D binary mask (same shape as volume)
              - volume_cc: Volume of the segmented structure in cubic centimeters (cc)
        """
        if not self.enabled:
            return np.zeros_like(volume, dtype=np.uint8), 0.0

        if self.onnx_session is not None:
            try:
                return self._run_onnx_inference(volume, spacing)
            except Exception as exc:
                logger.error(f"ONNX inference failed: {exc}. Falling back to Mock Engine.")

        return self._run_mock_segmentation(volume, modality, spacing)

    def _run_onnx_inference(self, volume: np.ndarray, spacing: tuple[float, float, float]) -> tuple[np.ndarray, float]:
        """Runs inference using the loaded ONNX session."""
        # Pad or resize to model's expected shape, e.g., (1, 1, D, H, W)
        input_data = np.expand_dims(np.expand_dims(volume, axis=0), axis=0).astype(np.float32)
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: input_data})
        probs = outputs[0][0, 0]  # squeeze batch and channel

        binary_mask = (probs >= self.threshold).astype(np.uint8)
        volume_cc = self._calculate_volume_cc(binary_mask, spacing)
        return binary_mask, volume_cc

    def _run_mock_segmentation(
        self,
        volume: np.ndarray,
        modality: str,
        spacing: tuple[float, float, float]
    ) -> tuple[np.ndarray, float]:
        """
        Simulates AI segmentation using morphological and intensity operations.
        - MRI (MR): segments a central butterfly-shaped structure (the ventricles).
        - CT: segments high-intensity bone region.
        - Other: segments a center-of-mass spherical lesion.
        """
        shape = volume.shape
        binary_mask = np.zeros(shape, dtype=np.uint8)

        if ndimage is None:
            # Fallback if scipy is somehow missing
            logger.warning("scipy not available for morphological mock segmentation. Returning center sphere.")
            z, y, x = shape
            cz, cy, cx = z // 2, y // 2, x // 2
            r = min(shape) // 6
            zs, ys, xs = np.ogrid[:z, :y, :x]
            dist = np.sqrt((zs - cz)**2 + (ys - cy)**2 + (xs - cx)**2)
            binary_mask[dist < r] = 1
            volume_cc = self._calculate_volume_cc(binary_mask, spacing)
            return binary_mask, volume_cc

        if modality == "MR":
            # Segment ventricles: typically lower-intensity cavities near the center of the brain
            # In a normalized MR (foreground_zscore), the ventricles have values < -0.4
            center_y, center_x = shape[1] // 2, shape[2] // 2
            
            # Create a butterfly-shaped ventricle seeding mask in the center slices
            z_start, z_end = int(shape[0] * 0.3), int(shape[0] * 0.7)
            seed = np.zeros(shape, dtype=bool)
            
            # Draw butterfly-like ventricles in coordinate space
            for z in range(z_start, z_end):
                # Scale lateral ventricle size depending on z slice
                scale = 1.0 - 0.5 * abs((z - (z_start + z_end)/2) / ((z_end - z_start)/2))
                r_y = int(shape[1] * 0.12 * scale)
                r_x = int(shape[2] * 0.08 * scale)
                
                # Left horn
                seed[z, center_y - r_y : center_y + r_y, center_x - r_x - int(r_x*0.5) : center_x - int(r_x*0.5)] = True
                # Right horn
                seed[z, center_y - r_y : center_y + r_y, center_x + int(r_x*0.5) : center_x + r_x + int(r_x*0.5)] = True

            # Intersect seed with lower-intensity voxels (ventricles are dark on T1w)
            # and exclude pure background (zeroes)
            candidate = (volume < -0.2) & (volume != 0.0)
            ventricles = candidate & seed
            
            # Morphological cleanup
            ventricles = ndimage.binary_closing(ventricles, iterations=1)
            ventricles = ndimage.binary_fill_holes(ventricles)
            binary_mask[ventricles] = 1

        elif modality == "CT":
            # Segment bone: high-density structures (e.g. HU > 250)
            # In CT windowing / min-max, it's the upper range.
            # Let's segment values above 0.7 of the max range or above 1.5 z-score
            bone_threshold = 1.2 if np.any(volume < 0) else 0.75
            candidate = volume > bone_threshold
            
            # Keep only the largest connected components (skeletal structure)
            labeled, num_features = ndimage.label(candidate)
            if num_features > 0:
                sizes = ndimage.sum(candidate, labeled, range(1, num_features + 1))
                largest_label = np.argmax(sizes) + 1
                binary_mask[labeled == largest_label] = 1

        else:
            # Simulate a generic lesion (spherical tumor in the center)
            z, y, x = shape
            cz, cy, cx = z // 2, y // 2, x // 2
            r = min(shape) // 8
            
            # Add some noise to make the border realistic
            zs, ys, xs = np.ogrid[:z, :y, :x]
            dist = np.sqrt((zs - cz)**2 + (ys - cy)**2 + (xs - cx)**2)
            noise = np.random.default_rng(42).normal(0, 1.5, shape)
            binary_mask[(dist + noise) < r] = 1

        volume_cc = self._calculate_volume_cc(binary_mask, spacing)
        return binary_mask, volume_cc

    def _calculate_volume_cc(self, mask: np.ndarray, spacing: tuple[float, float, float]) -> float:
        """Convert voxel count to volume in cubic centimeters (cc)."""
        voxel_count = int(np.sum(mask))
        voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
        return round((voxel_count * voxel_vol_mm3) / 1000.0, 2)
