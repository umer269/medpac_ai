"""
tests/test_segmenter.py
───────────────────────
Unit tests for the AI Segmenter module (Mock and ONNX modes).
"""

from __future__ import annotations

import numpy as np

from etl.segmenter import DicomSegmenter


def test_mock_mri_segmentation():
    # Generate mock T1w brain MRI volume (foreground foreground z-score normalized)
    # Background is 0, ventricles are low (e.g. -0.5), brain tissues are positive
    shape = (20, 48, 48)
    vol = np.zeros(shape, dtype=np.float32)
    # Central cavity region (ventricles candidate)
    vol[6:14, 20:28, 20:28] = -0.5

    segmenter = DicomSegmenter(enabled=True, model_type="mock", target_structure="ventricles")
    spacing = (1.0, 1.0, 1.0)

    mask, vol_cc = segmenter.segment(vol, "MR", spacing)

    assert mask.shape == shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})
    assert vol_cc >= 0.0


def test_mock_ct_segmentation():
    # Generate mock CT volume
    shape = (10, 32, 32)
    vol = np.zeros(shape, dtype=np.float32)
    # High-intensity bone structure (e.g., standard scale > 0.8)
    vol[4:7, 12:20, 12:20] = 0.9

    segmenter = DicomSegmenter(enabled=True, model_type="mock", target_structure="bones")
    spacing = (1.5, 1.5, 2.0)

    mask, vol_cc = segmenter.segment(vol, "CT", spacing)

    assert mask.shape == shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})
    assert vol_cc > 0.0  # Should segment the block

    # Calculate exact voxel count * voxel volume / 1000.0
    voxel_vol = 1.5 * 1.5 * 2.0
    expected_cc = round((3 * 8 * 8 * voxel_vol) / 1000.0, 2)
    assert abs(vol_cc - expected_cc) < 0.05


def test_segmentation_disabled():
    shape = (5, 10, 10)
    vol = np.ones(shape, dtype=np.float32)
    segmenter = DicomSegmenter(enabled=False)
    mask, vol_cc = segmenter.segment(vol, "MR", (1.0, 1.0, 1.0))

    assert np.all(mask == 0)
    assert vol_cc == 0.0


def test_onnx_fallback_to_mock():
    # Specifying ONNX mode without library or file should fallback to mock gracefully
    shape = (10, 16, 16)
    vol = np.ones(shape, dtype=np.float32)
    segmenter = DicomSegmenter(enabled=True, model_type="onnx", model_path="nonexistent.onnx")

    assert segmenter.onnx_session is None
    mask, vol_cc = segmenter.segment(vol, "MR", (1.0, 1.0, 1.0))

    assert mask.shape == shape
    assert mask.dtype == np.uint8
