"""
tests/test_transformer.py
──────────────────────────
Unit tests for the Transform stage.
Uses synthetic numpy arrays so no real DICOM files are required.
"""

from __future__ import annotations

import numpy as np
import pytest

from etl.models      import DicomSeries
from etl.transformer import (
    DicomTransformer,
    _normalize_min_max,
    _normalize_z_score,
    _normalize_window_level,
    _extract_preview_slice,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_series(shape=(10, 64, 64), pixel_spacing=(1.0, 1.0)) -> DicomSeries:
    """Return a minimal synthetic DicomSeries for testing."""
    return DicomSeries(
        series_uid      = "1.2.3.4.5",
        study_uid       = "1.2.3.4",
        patient_id      = "TEST001",
        patient_name    = "Test^Patient",
        modality        = "MR",
        study_date      = "20240101",
        study_desc      = "Test scan",
        num_slices      = shape[0],
        rows            = shape[1],
        cols            = shape[2],
        pixel_spacing   = pixel_spacing,
        slice_thickness = 2.5,
        pixel_array     = np.random.uniform(-200, 1000, shape).astype(np.float32),
    )


# ─── Normalization tests ──────────────────────────────────────────────────────

class TestNormalizationFunctions:

    def test_z_score_mean_and_std(self):
        arr = np.random.normal(loc=500, scale=100, size=(20, 64, 64)).astype(np.float32)
        result = _normalize_z_score(arr)
        assert abs(result.mean()) < 1e-5, "Z-score mean should be ~0"
        assert abs(result.std() - 1.0) < 1e-5, "Z-score std should be ~1"

    def test_min_max_range(self):
        arr = np.random.uniform(-1000, 3000, (10, 32, 32)).astype(np.float32)
        result = _normalize_min_max(arr)
        assert result.min() >= 0.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_min_max_constant_array(self):
        """Constant arrays should return zeros without division-by-zero."""
        arr = np.full((5, 5, 5), 42.0, dtype=np.float32)
        result = _normalize_min_max(arr)
        assert (result == 0).all()

    def test_window_level_clips(self):
        arr = np.array([-200.0, 0.0, 40.0, 300.0, 1000.0])
        # WC=40, WW=400 → window = [-160, 240]
        result = _normalize_window_level(arr, window_center=40, window_width=400)
        assert result.min() >= 0.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6
        # Values below the window → 0; above → 1
        assert result[0]  == pytest.approx(0.0, abs=1e-4)
        assert result[-1] == pytest.approx(1.0, abs=1e-4)


# ─── Preview slice tests ──────────────────────────────────────────────────────

class TestPreviewSlice:

    def test_preview_shape_axis0(self):
        vol   = np.random.rand(20, 64, 128).astype(np.float32)
        preview = _extract_preview_slice(vol, axis=0)
        assert preview.shape == (64, 128)

    def test_preview_shape_axis2(self):
        vol     = np.random.rand(20, 64, 128).astype(np.float32)
        preview = _extract_preview_slice(vol, axis=2)
        assert preview.shape == (20, 64)

    def test_preview_dtype_uint8(self):
        vol     = np.random.rand(10, 32, 32).astype(np.float32)
        preview = _extract_preview_slice(vol)
        assert preview.dtype == np.uint8


# ─── DicomTransformer integration tests ───────────────────────────────────────

class TestDicomTransformer:

    def test_transform_returns_correct_type(self):
        from etl.models import TransformedSeries
        series      = make_series()
        transformer = DicomTransformer(normalization="min_max", save_preview=True)
        result      = transformer.transform(series)
        assert isinstance(result, TransformedSeries)

    def test_transform_preserves_shape_without_resampling(self):
        shape  = (12, 64, 64)
        series = make_series(shape=shape)
        transformer = DicomTransformer(target_spacing=None, save_preview=False)
        result = transformer.transform(series)
        assert result.normalized_array.shape == shape

    def test_affine_is_4x4(self):
        series      = make_series()
        transformer = DicomTransformer()
        result      = transformer.transform(series)
        assert result.affine.shape == (4, 4)

    def test_unknown_normalization_raises(self):
        series      = make_series()
        transformer = DicomTransformer(normalization="does_not_exist")
        with pytest.raises(ValueError, match="Unknown normalization"):
            transformer.transform(series)

    def test_z_score_output_approximately_unit_normal(self):
        shape  = (20, 64, 64)
        series = make_series(shape=shape)
        transformer = DicomTransformer(
            normalization="z_score",
            clip_percentile_low=0.0,
            clip_percentile_high=100.0,
            target_spacing=None,
            save_preview=False,
        )
        result = transformer.transform(series)
        arr = result.normalized_array
        assert abs(arr.mean()) < 0.05
        assert abs(arr.std()  - 1.0) < 0.05
