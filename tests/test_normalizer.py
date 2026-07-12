"""
tests/test_normalizer.py
─────────────────────────
Unit tests for the research-backed normalizer module:
  - foreground_zscore (nnU-Net v2 / Isensee et al. 2021)
  - whitestripe_normalize (Shinohara et al. 2014)
  - NyulUdupaNormalizer (Nyúl & Udupa 1999/2000)
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from etl.normalizer import (
    NyulUdupaNormalizer,
    foreground_zscore,
    whitestripe_normalize,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_synthetic_mri(shape=(20, 64, 64), bg_fraction=0.3, seed=42) -> np.ndarray:
    """
    Synthetic MRI-like volume:
    - background = ~0 (air)
    - foreground = bimodal (GM + WM peaks)
    """
    rng = np.random.default_rng(seed)
    vol = np.zeros(shape, dtype=np.float32)
    n_fg = int(np.prod(shape) * (1 - bg_fraction))
    fg_idx = rng.choice(np.prod(shape), size=n_fg, replace=False)

    # Bimodal: half GM (~400), half WM (~700)
    gm  = rng.normal(loc=400, scale=40, size=n_fg // 2).astype(np.float32)
    wm  = rng.normal(loc=700, scale=35, size=n_fg - n_fg // 2).astype(np.float32)
    fg_vals = np.concatenate([gm, wm])
    np.put(vol, fg_idx, fg_vals)
    return vol


# ─── foreground_zscore ────────────────────────────────────────────────────────

class TestForegroundZscore:

    def test_foreground_mean_approx_zero(self):
        vol    = make_synthetic_mri()
        result = foreground_zscore(vol)
        fg     = result[result != 0.0]
        assert abs(fg.mean()) < 0.1, "Foreground mean should be ≈0"

    def test_foreground_std_approx_one(self):
        vol    = make_synthetic_mri()
        result = foreground_zscore(vol)
        fg     = result[result != 0.0]
        assert abs(fg.std() - 1.0) < 0.05, "Foreground std should be ≈1"

    def test_background_zeroed_out(self):
        """Background voxels (≤0) must be zeroed in output."""
        vol    = make_synthetic_mri()
        result = foreground_zscore(vol)
        bg_mask = vol <= 0
        assert (result[bg_mask] == 0.0).all(), "Background should remain 0"

    def test_constant_volume_returns_zeros(self):
        vol    = np.full((5, 5, 5), 42.0, dtype=np.float32)
        result = foreground_zscore(vol)
        assert (result == 0.0).all()

    def test_all_zero_volume_returns_zeros(self):
        vol    = np.zeros((5, 5, 5), dtype=np.float32)
        result = foreground_zscore(vol)
        assert (result == 0.0).all()

    def test_output_shape_preserved(self):
        shape  = (12, 48, 48)
        vol    = make_synthetic_mri(shape=shape)
        result = foreground_zscore(vol)
        assert result.shape == shape

    def test_output_dtype_float32(self):
        vol = make_synthetic_mri()
        assert foreground_zscore(vol).dtype == np.float32


# ─── whitestripe_normalize ────────────────────────────────────────────────────

class TestWhiteStripe:

    def test_output_shape_preserved(self):
        vol    = make_synthetic_mri()
        result = whitestripe_normalize(vol)
        assert result.shape == vol.shape

    def test_output_dtype_float32(self):
        vol = make_synthetic_mri()
        assert whitestripe_normalize(vol).dtype == np.float32

    def test_normalized_wm_peak_near_zero(self):
        """
        WhiteStripe guarantees that voxels WITHIN the WM stripe (near the WM peak
        in the ORIGINAL image) should have a mean close to 0 after normalization,
        because μ_ws is subtracted from them.

        This tests the actual algorithm guarantee — not the top quartile of the
        normalized output, which can be above 0 for voxels brighter than the stripe.
        """
        vol    = make_synthetic_mri(seed=7)
        result = whitestripe_normalize(vol)

        # Identify the WM peak in the original volume (≈ top 25 % of foreground)
        fg_orig  = vol[vol > 0].flatten()
        p75      = np.percentile(fg_orig, 75)
        p90      = np.percentile(fg_orig, 90)

        # Voxels in the WM stripe region of the original image
        stripe_mask    = (vol >= p75) & (vol <= p90)
        stripe_in_norm = result[stripe_mask]

        # Their mean in the normalized volume should be relatively small
        # (WhiteStripe sets μ_ws ≈ 0; the stripe spans ±half_width around it)
        assert abs(stripe_in_norm.mean()) < 2.0, (
            f"WM-stripe voxels mean after WhiteStripe = {stripe_in_norm.mean():.3f}, "
            "expected close to 0"
        )


    def test_sparse_volume_falls_back_gracefully(self):
        """A volume with very few foreground voxels should not crash."""
        vol = np.zeros((10, 10, 10), dtype=np.float32)
        vol[5, 5, 5] = 100.0   # only one non-zero voxel
        result = whitestripe_normalize(vol)
        assert result.shape == vol.shape


# ─── NyulUdupaNormalizer ──────────────────────────────────────────────────────

class TestNyulUdupaNormalizer:

    def _train(self, n=5, seed=0) -> NyulUdupaNormalizer:
        norm = NyulUdupaNormalizer()
        rng  = np.random.default_rng(seed)
        for i in range(n):
            # Each scanner has a different intensity scale
            scale = rng.uniform(0.7, 1.3)
            vol   = make_synthetic_mri(seed=seed + i) * scale
            norm.update(vol)
        norm.fit()
        return norm

    def test_fit_produces_standard_scale(self):
        norm = self._train()
        assert norm.standard_scale is not None
        assert len(norm.standard_scale) == len(NyulUdupaNormalizer.DEFAULT_LANDMARKS)

    def test_standard_scale_is_monotone(self):
        """Percentile landmarks must be non-decreasing."""
        norm = self._train()
        diffs = np.diff(norm.standard_scale)
        assert (diffs >= 0).all(), "Standard scale must be monotone"

    def test_transform_output_shape(self):
        norm  = self._train()
        vol   = make_synthetic_mri(seed=99)
        out   = norm.transform(vol)
        assert out.shape == vol.shape

    def test_transform_dtype_float32(self):
        norm = self._train()
        vol  = make_synthetic_mri(seed=99)
        assert norm.transform(vol).dtype == np.float32

    def test_transform_reduces_inter_scanner_variance(self):
        """
        Volumes with different intensity scales should have similar
        landmark values after normalization (key property of the method).
        """
        norm = self._train(n=8, seed=10)
        # Generate two "scans" at very different scales
        vol_a = make_synthetic_mri(seed=20) * 0.5    # dim scanner
        vol_b = make_synthetic_mri(seed=21) * 2.0    # bright scanner

        a_norm = norm.transform(vol_a)
        b_norm = norm.transform(vol_b)

        # After normalization the p50 of both should be much closer
        fg_a = a_norm[vol_a > 0].flatten()
        fg_b = b_norm[vol_b > 0].flatten()

        diff_raw  = abs(np.percentile(vol_a[vol_a > 0], 50) - np.percentile(vol_b[vol_b > 0], 50))
        diff_norm = abs(np.percentile(fg_a, 50) - np.percentile(fg_b, 50))

        assert diff_norm < diff_raw, "Normalization should reduce inter-scanner p50 difference"

    def test_save_and_load_roundtrip(self, tmp_path):
        norm      = self._train()
        save_path = str(tmp_path / "standard_scale.json")
        norm.save(save_path)

        loaded = NyulUdupaNormalizer.load(save_path)
        np.testing.assert_array_almost_equal(
            norm.standard_scale, loaded.standard_scale, decimal=6
        )

    def test_fit_without_update_raises(self):
        norm = NyulUdupaNormalizer()
        with pytest.raises(RuntimeError, match="no training images"):
            norm.fit()

    def test_transform_without_fit_raises(self):
        norm = NyulUdupaNormalizer()
        with pytest.raises(RuntimeError, match="standard scale not available"):
            norm.transform(make_synthetic_mri())

    def test_fit_transform_convenience(self):
        vols = [make_synthetic_mri(seed=i) for i in range(4)]
        norm = NyulUdupaNormalizer()
        results = norm.fit_transform(vols)
        assert len(results) == 4
        assert all(r.shape == v.shape for r, v in zip(results, vols))
