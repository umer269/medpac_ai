"""
etl/normalizer.py
─────────────────────────────────────────────────────────────────────────────
Advanced normalization strategies backed by peer-reviewed research.

Three strategies are implemented here — plug directly into the Transformer:

1.  foreground_zscore  — nnU-Net v2 strategy (Isensee et al., Nature Methods 2021 / 2024)
    Computes z-score statistics ONLY from foreground (non-zero) voxels.
    Much more robust than whole-volume z-score because background air/padding
    voxels corrupt the mean and std.

2.  whitestripe        — Shinohara et al. (2014), actively maintained 2024.
    Tissue-anchored normalization: finds the white matter (WM) peak in the
    intensity histogram via KDE + peak detection, then z-scores relative to
    the "white matter stripe" (voxels near the WM peak).
    Ideal for T1-weighted MRI — robust across scanners.

3.  NyulUdupaNormalizer — Nyúl & Udupa (1999, 2000), still gold standard 2024.
    Population-level landmark normalization: fits a set of percentile
    "landmarks" across a training set and maps each new scan onto the
    standard scale via piecewise-linear interpolation.
    Used by Siemens Healthineers and major neuro-imaging consortia.

References:
  - Isensee F. et al. (2021) "nnU-Net: a self-configuring method for deep
    learning-based biomedical image segmentation" Nature Methods 18, 203–211.
  - Isensee F. et al. (2024) "nnU-Net revisited: A call for rigorous
    validation in 3D medical image segmentation" arXiv:2404.09556
  - Shinohara R.T. et al. (2014) "Statistical normalization techniques for
    magnetic resonance imaging" NeuroImage: Clinical 6, 9–19.
  - Nyúl L.G. & Udupa J.K. (1999) "On standardizing the MR image intensity
    scale" Magnetic Resonance in Medicine 42(6), 1072–1081.
  - Nyúl L.G., Udupa J.K. & Zhang X. (2000) "New variants of a method of
    MRI scale standardization" IEEE TMI 19(2), 143–150.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


# ─────────────────────────────────────────────────────────────────────────────
# 1.  nnU-Net Foreground Z-Score  (Isensee et al., 2021 / 2024)
# ─────────────────────────────────────────────────────────────────────────────

def foreground_zscore(
    volume: np.ndarray,
    foreground_threshold: float = 0.0,
) -> np.ndarray:
    """
    Z-score normalization using only foreground (non-zero) voxel statistics.

    Rationale (nnU-Net v2):
        Whole-volume z-score is corrupted by the large fraction of background
        (air/padding) voxels whose near-zero intensities pull the mean toward
        zero and inflate the standard deviation.  Computing stats over
        foreground only gives a physiologically meaningful scale.

    Args:
        volume:               3-D float numpy array (Z, Y, X)
        foreground_threshold: Voxels ABOVE this value are treated as foreground.
                              Default 0.0 matches nnU-Net's "nonzero" mask.

    Returns:
        Normalized float32 array with the same shape as `volume`.
        Background voxels are set to 0 after normalization (they carry
        no anatomical signal and would otherwise be mapped to a non-zero value).
    """
    volume  = volume.astype(np.float64)
    fg_mask = volume > foreground_threshold
    fg_vals = volume[fg_mask]

    if fg_vals.size == 0:
        logger.warning("foreground_zscore: no foreground voxels found — returning zeros.")
        return np.zeros_like(volume, dtype=np.float32)

    fg_mean = fg_vals.mean()
    fg_std  = fg_vals.std()

    if fg_std < 1e-8:
        logger.warning("foreground_zscore: foreground std ≈ 0 — returning zeros.")
        return np.zeros_like(volume, dtype=np.float32)

    normalized         = (volume - fg_mean) / fg_std
    normalized[~fg_mask] = 0.0   # zero out background — no anatomical info
    return normalized.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  WhiteStripe  (Shinohara et al., 2014)
# ─────────────────────────────────────────────────────────────────────────────

def whitestripe_normalize(
    volume:        np.ndarray,
    mask:          Optional[np.ndarray] = None,
    stripe_width:  float = 0.05,   # fraction of value range around WM peak
    n_peaks:       int   = 3,      # number of histogram peaks to detect
    wm_peak_rank:  int   = -1,     # index into sorted peaks (-1 = rightmost = WM for T1w)
    bw_method:     str   = "silverman",
) -> np.ndarray:
    """
    WhiteStripe intensity normalization for T1-weighted MRI.

    Algorithm (Shinohara et al., 2014 — NeuroImage: Clinical):
      1. Smooth the foreground intensity histogram using a Gaussian KDE.
      2. Locate histogram peaks; select the rightmost peak as the
         White Matter (WM) intensity mode for T1w scans.
      3. Define a "stripe" of voxels near the WM peak.
      4. Compute stripe mean (μ_ws) and std (σ_ws).
      5. Normalize the entire volume: (v - μ_ws) / σ_ws.

    Args:
        volume:        3-D float numpy array.
        mask:          Optional brain mask (same shape). Only unmasked voxels
                       used for WM peak estimation.
        stripe_width:  Half-width of the stripe as a fraction of the value
                       range (default 5 %).
        n_peaks:       Maximum number of histogram peaks to search for.
        wm_peak_rank:  Which peak to use as WM mode (-1 = rightmost for T1w;
                       use 0 for leftmost = CSF, 1 = GM for T2w).
        bw_method:     KDE bandwidth estimation method.

    Returns:
        Normalized float32 array, same shape as input.
    """
    volume = volume.astype(np.float64)

    # Select analysis voxels
    if mask is not None:
        vals = volume[mask > 0].flatten()
    else:
        vals = volume[volume > 0].flatten()

    if len(vals) < 100:
        logger.warning("WhiteStripe: insufficient foreground voxels — falling back to z-score.")
        return foreground_zscore(volume)

    # ── 1. KDE smoothing of intensity histogram ───────────────────────────────
    kde        = gaussian_kde(vals, bw_method=bw_method)
    x_range    = np.linspace(vals.min(), vals.max(), 2000)
    density    = kde(x_range)

    # ── 2. Peak detection ─────────────────────────────────────────────────────
    peaks, properties = find_peaks(
        density,
        prominence = density.max() * 0.02,   # at least 2% of max prominence
        distance   = len(x_range) // 20,     # min distance between peaks
    )

    if len(peaks) == 0:
        logger.warning("WhiteStripe: no peaks detected — falling back to foreground z-score.")
        return foreground_zscore(volume)

    # Sort peaks by prominence (descending) and select the WM peak
    prominences = properties["prominences"]
    sorted_by_prominence = peaks[np.argsort(prominences)[::-1]]
    top_peaks = sorted(sorted_by_prominence[:n_peaks].tolist())   # sort by x position

    wm_peak_x = x_range[top_peaks[wm_peak_rank]]
    logger.debug(f"WhiteStripe: WM peak intensity = {wm_peak_x:.2f}")

    # ── 3. Define stripe ─────────────────────────────────────────────────────
    val_range  = vals.max() - vals.min()
    half_width = stripe_width * val_range
    stripe_lo  = wm_peak_x - half_width
    stripe_hi  = wm_peak_x + half_width

    stripe_vals = vals[(vals >= stripe_lo) & (vals <= stripe_hi)]

    if len(stripe_vals) < 10:
        logger.warning("WhiteStripe: stripe too narrow (< 10 voxels) — widening to ±10 %.")
        half_width  = 0.10 * val_range
        stripe_vals = vals[(vals >= wm_peak_x - half_width) & (vals <= wm_peak_x + half_width)]

    # ── 4. Stripe statistics ──────────────────────────────────────────────────
    mu_ws    = float(stripe_vals.mean())
    sigma_ws = float(stripe_vals.std())

    if sigma_ws < 1e-8:
        logger.warning("WhiteStripe: stripe std ≈ 0 — returning foreground z-score.")
        return foreground_zscore(volume)

    # ── 5. Normalize ──────────────────────────────────────────────────────────
    normalized = (volume - mu_ws) / sigma_ws
    return normalized.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Nyúl–Udupa Histogram Landmark Normalization  (1999 / 2000)
# ─────────────────────────────────────────────────────────────────────────────

class NyulUdupaNormalizer:
    """
    Population-level piecewise-linear MRI intensity standardization.

    Usage:
        # Training phase (once across a representative dataset)
        norm = NyulUdupaNormalizer()
        for series in training_series:
            norm.update(series.pixel_array)
        norm.fit()
        norm.save("standard_scale.json")

        # Inference phase (for each new scan)
        norm = NyulUdupaNormalizer.load("standard_scale.json")
        normalized_volume = norm.transform(new_volume)

    Algorithm (Nyúl & Udupa 1999; Nyúl, Udupa & Zhang 2000):
        Training:
            1. For each training image, compute M percentile landmarks
               (default p1, p10, p20, … p90, p99) from foreground voxels.
            2. Average the landmarks across all training images → standard scale.

        Transformation:
            For a new image T:
            1. Compute T's percentile landmarks.
            2. Build a piecewise-linear mapping:
                   T_norm(x) = s_k + (x - l_k) × (s_{k+1} - s_k) / (l_{k+1} - l_k)
               where s_k = standard landmark k, l_k = image landmark k.
            3. Linear extrapolation outside [l_p1, l_p99].
    """

    DEFAULT_LANDMARKS: List[float] = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]

    def __init__(self, landmarks: Optional[List[float]] = None):
        self.landmarks:       List[float] = landmarks or self.DEFAULT_LANDMARKS
        self._sum_landmarks:  Optional[np.ndarray] = None
        self._n_images:       int = 0
        self.standard_scale:  Optional[np.ndarray] = None   # set after fit()

    # ── Training API ──────────────────────────────────────────────────────────

    def update(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        """Accumulate percentile statistics from one training volume."""
        vals = volume[mask > 0].flatten() if mask is not None else volume[volume > 0].flatten()

        if len(vals) < 100:
            logger.warning("NyulUdupa.update: skipping volume with < 100 foreground voxels.")
            return

        lm = np.percentile(vals, self.landmarks)

        if self._sum_landmarks is None:
            self._sum_landmarks = np.zeros(len(self.landmarks), dtype=np.float64)
        self._sum_landmarks += lm
        self._n_images += 1

    def fit(self) -> "NyulUdupaNormalizer":
        """
        Compute the standard scale = mean landmark values across all training images.
        Must be called after all `.update()` calls.
        """
        if self._n_images == 0:
            raise RuntimeError("NyulUdupa.fit(): no training images were added via .update().")

        self.standard_scale = self._sum_landmarks / self._n_images
        logger.info(
            f"NyulUdupa: fitted on {self._n_images} images.  "
            f"Standard scale (p1→p99): "
            f"[{self.standard_scale[0]:.2f} … {self.standard_scale[-1]:.2f}]"
        )
        return self

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the standard scale to a JSON file."""
        if self.standard_scale is None:
            raise RuntimeError("Call .fit() before saving.")
        data = {
            "landmarks":      self.landmarks,
            "standard_scale": self.standard_scale.tolist(),
            "n_training_images": self._n_images,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        logger.info(f"NyulUdupa: saved standard scale → {path}")

    @classmethod
    def load(cls, path: str) -> "NyulUdupaNormalizer":
        """Load a pre-fitted standard scale from a JSON file."""
        with open(path) as fh:
            data = json.load(fh)
        obj = cls(landmarks=data["landmarks"])
        obj.standard_scale = np.array(data["standard_scale"])
        obj._n_images      = data.get("n_training_images", 0)
        logger.info(f"NyulUdupa: loaded standard scale from '{path}' "
                    f"(trained on {obj._n_images} images).")
        return obj

    # ── Transform API ─────────────────────────────────────────────────────────

    def transform(
        self,
        volume: np.ndarray,
        mask:   Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply piecewise-linear landmark normalization to a new volume.

        Args:
            volume: 3-D float numpy array.
            mask:   Optional foreground mask. If None, non-zero voxels are used.

        Returns:
            Normalized float32 array, same shape as input.

        Raises:
            RuntimeError: If the standard scale has not been fitted or loaded.
        """
        if self.standard_scale is None:
            raise RuntimeError(
                "NyulUdupa.transform(): standard scale not available.  "
                "Call .fit() or .load() first."
            )

        volume = volume.astype(np.float64)
        vals   = (
            volume[mask > 0].flatten()
            if mask is not None
            else volume[volume > 0].flatten()
        )

        if len(vals) < 100:
            logger.warning("NyulUdupa.transform: too few foreground voxels — returning raw.")
            return volume.astype(np.float32)

        # Compute this image's landmarks
        img_landmarks = np.percentile(vals, self.landmarks)

        # Build piecewise-linear mapping: img_landmarks → standard_scale
        # scipy.interpolate.interp1d with fill_value extrapolation handles
        # values outside the [p1, p99] range automatically.
        if len(np.unique(img_landmarks)) < 2:
            logger.warning("NyulUdupa.transform: degenerate landmarks — returning foreground z-score.")
            return foreground_zscore(volume)

        mapping = interp1d(
            img_landmarks,
            self.standard_scale,
            kind        = "linear",
            bounds_error = False,
            fill_value   = "extrapolate",
        )

        normalized = mapping(volume)
        return normalized.astype(np.float32)

    # ── Convenience: fit + transform in one step ──────────────────────────────

    def fit_transform(
        self,
        volumes: List[np.ndarray],
        masks:   Optional[List[Optional[np.ndarray]]] = None,
    ) -> List[np.ndarray]:
        """
        Fit the standard scale on all `volumes` and immediately return
        the normalized versions. Useful for single-dataset pipelines.
        """
        if masks is None:
            masks = [None] * len(volumes)
        for vol, msk in zip(volumes, masks):
            self.update(vol, msk)
        self.fit()
        return [self.transform(vol, msk) for vol, msk in zip(volumes, masks)]
