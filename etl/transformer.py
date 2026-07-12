"""
etl/transformer.py
──────────────────
TRANSFORM stage — normalizes pixel intensities, resamples voxel spacing,
builds a NIfTI affine, and optionally generates a PNG preview slice.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from loguru import logger
from scipy.ndimage import zoom    # type: ignore[import]

from etl.models     import DicomSeries, TransformedSeries
from etl.normalizer import foreground_zscore, whitestripe_normalize


# ─── Normalization strategies ─────────────────────────────────────────────────

def _normalize_z_score(arr: np.ndarray) -> np.ndarray:
    """
    Zero-mean, unit-variance normalization across the entire volume.
    Small epsilon avoids division-by-zero for empty/constant volumes.
    """
    mean = arr.mean()
    std  = arr.std() + 1e-8
    return (arr - mean) / std


def _normalize_min_max(arr: np.ndarray) -> np.ndarray:
    """Scales intensities to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _normalize_window_level(
    arr: np.ndarray,
    window_center: float,
    window_width: float,
) -> np.ndarray:
    """
    Standard radiology window/level mapping used for CT Hounsfield Units.
    Output is clipped to [0, 1].
    """
    lo = window_center - window_width / 2.0
    hi = window_center + window_width / 2.0
    arr = np.clip(arr, lo, hi)
    return (arr - lo) / (hi - lo)


# ─── Resampling ───────────────────────────────────────────────────────────────

def _resample_volume(
    volume: np.ndarray,
    current_spacing: Tuple[float, float, float],
    target_spacing:  Tuple[float, float, float],
) -> np.ndarray:
    """
    Resamples a (Z, Y, X) volume from current_spacing to target_spacing (mm).
    Uses tri-linear interpolation (order=1) to preserve intensity values.
    """
    zoom_factors = tuple(
        cs / ts for cs, ts in zip(current_spacing, target_spacing)
    )
    logger.debug(f"Resampling with zoom factors: {zoom_factors}")
    return zoom(volume, zoom_factors, order=1, prefilter=False)


# ─── Affine construction ──────────────────────────────────────────────────────

def _build_simple_affine(
    pixel_spacing: Tuple[float, float],
    slice_thickness: float,
) -> np.ndarray:
    """
    Build a minimal NIfTI affine from pixel spacing and slice thickness.
    Assumes standard RAS orientation (scanner-space identity rotation).
    """
    affine        = np.eye(4, dtype=float)
    affine[0, 0]  = pixel_spacing[1]   # column spacing → x
    affine[1, 1]  = pixel_spacing[0]   # row spacing    → y
    affine[2, 2]  = slice_thickness    # slice spacing  → z
    return affine


# ─── Preview slice ────────────────────────────────────────────────────────────

def _extract_preview_slice(
    volume: np.ndarray,
    axis: int = 2,
) -> np.ndarray:
    """
    Returns the middle slice along the given axis as a 2-D float32 array
    scaled to [0, 255] uint8 for PNG export.
    """
    idx = volume.shape[axis] // 2
    if axis == 0:
        sl = volume[idx, :, :]
    elif axis == 1:
        sl = volume[:, idx, :]
    else:
        sl = volume[:, :, idx]

    lo, hi = sl.min(), sl.max()
    if hi - lo < 1e-8:
        return np.zeros(sl.shape, dtype=np.uint8)
    return ((sl - lo) / (hi - lo) * 255).astype(np.uint8)


# ─── Transformer class ────────────────────────────────────────────────────────

class DicomTransformer:
    """
    Applies configurable preprocessing to a DicomSeries and returns a
    TransformedSeries ready to be saved as NIfTI.
    """

    def __init__(
        self,
        normalization:        str   = "z_score",
        window_center:        float = 40.0,
        window_width:         float = 400.0,
        target_spacing:       Optional[Tuple[float, float, float]] = None,
        clip_percentile_low:  float = 0.5,
        clip_percentile_high: float = 99.5,
        save_preview:         bool  = True,
        preview_slice_axis:   int   = 2,
    ):
        self.normalization        = normalization
        self.window_center        = window_center
        self.window_width         = window_width
        self.target_spacing       = target_spacing
        self.clip_percentile_low  = clip_percentile_low
        self.clip_percentile_high = clip_percentile_high
        self.save_preview         = save_preview
        self.preview_slice_axis   = preview_slice_axis

    # ── public API ────────────────────────────────────────────────────────────

    def transform(self, series: DicomSeries) -> TransformedSeries:
        logger.info(
            f"[Transform] {series.series_uid[:16]}… "
            f"shape={series.pixel_array.shape} modality={series.modality}"
        )

        volume = series.pixel_array.astype(np.float32).copy()

        # 1. Clip intensity outliers
        lo_val = np.percentile(volume, self.clip_percentile_low)
        hi_val = np.percentile(volume, self.clip_percentile_high)
        volume = np.clip(volume, lo_val, hi_val)
        logger.debug(f"Clipped intensity range: [{lo_val:.2f}, {hi_val:.2f}]")

        # 2. Optional resampling
        current_spacing = (
            series.slice_thickness,
            series.pixel_spacing[0],
            series.pixel_spacing[1],
        )
        if self.target_spacing is not None:
            volume = _resample_volume(volume, current_spacing, self.target_spacing)
            effective_spacing = self.target_spacing
        else:
            effective_spacing = current_spacing

        # 3. Normalization
        if self.normalization == "z_score":
            volume = _normalize_z_score(volume)
        elif self.normalization == "min_max":
            volume = _normalize_min_max(volume)
        elif self.normalization == "window_level":
            volume = _normalize_window_level(volume, self.window_center, self.window_width)
        elif self.normalization == "foreground_zscore":
            # nnU-Net v2 strategy — Isensee et al., Nature Methods 2021/2024
            volume = foreground_zscore(volume)
        elif self.normalization == "whitestripe":
            # Shinohara et al. (2014) — tissue-anchored WM peak normalization
            volume = whitestripe_normalize(volume)
        else:
            raise ValueError(f"Unknown normalization strategy: {self.normalization!r}")

        logger.debug(
            f"Post-transform volume shape={volume.shape} "
            f"min={volume.min():.4f} max={volume.max():.4f}"
        )

        # 4. Build affine
        affine = _build_simple_affine(
            (effective_spacing[1], effective_spacing[2]),
            effective_spacing[0],
        )

        # 5. Optional preview slice
        preview = (
            _extract_preview_slice(volume, self.preview_slice_axis)
            if self.save_preview
            else None
        )

        return TransformedSeries(
            source            = series,
            normalized_array  = volume,
            affine            = affine,
            preview_slice     = preview,
        )
