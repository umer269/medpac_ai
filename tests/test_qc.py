"""
tests/test_qc.py
──────────────────
Unit tests for the MRIQC-inspired automated QC module.
"""

from __future__ import annotations

import numpy as np

from etl.models import DicomSeries
from etl.qc import QCReport, SeriesQCChecker

# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_series(pixel_array=None, num_slices=20) -> DicomSeries:
    if pixel_array is None:
        rng = np.random.default_rng(0)
        pixel_array = rng.normal(loc=500, scale=150, size=(num_slices, 64, 64)).astype(np.float32)
        pixel_array[pixel_array < 0] = 0.0  # simulate background

    return DicomSeries(
        series_uid="1.2.3.4.5",
        study_uid="1.2.3",
        patient_id="P001",
        patient_name="Doe^John",
        modality="MR",
        study_date="20240601",
        study_desc="Test MRI",
        num_slices=num_slices,
        rows=pixel_array.shape[1],
        cols=pixel_array.shape[2],
        pixel_spacing=(1.0, 1.0),
        slice_thickness=3.0,
        pixel_array=pixel_array,
    )


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestSeriesQCChecker:
    def test_good_series_passes(self):
        checker = SeriesQCChecker(min_slices=4, min_snr=1.0, min_fber=0.1)
        series = make_series()
        report = checker.check(series)
        assert isinstance(report, QCReport)
        assert report.passed, f"Expected pass, got issues: {report.issues}"

    def test_too_few_slices_fails(self):
        checker = SeriesQCChecker(min_slices=10)
        series = make_series(
            num_slices=3, pixel_array=np.random.rand(3, 64, 64).astype(np.float32) * 500
        )
        report = checker.check(series)
        assert not report.passed
        assert any("TOO_FEW_SLICES" in i for i in report.issues)

    def test_constant_image_flagged(self):
        checker = SeriesQCChecker()
        vol = np.full((20, 64, 64), 42.0, dtype=np.float32)
        series = make_series(pixel_array=vol)
        report = checker.check(series)
        assert any("CONSTANT_IMAGE" in i for i in report.issues)

    def test_report_contains_snr(self):
        checker = SeriesQCChecker(min_snr=0.0)  # disable SNR gate
        series = make_series()
        report = checker.check(series)
        assert report.snr > 0.0

    def test_report_contains_fber(self):
        checker = SeriesQCChecker(min_fber=0.0)
        series = make_series()
        report = checker.check(series)
        assert report.fber >= 0.0

    def test_report_contains_efc(self):
        checker = SeriesQCChecker()
        series = make_series()
        report = checker.check(series)
        assert isinstance(report.efc, float)

    def test_report_contains_cjv(self):
        checker = SeriesQCChecker()
        series = make_series()
        report = checker.check(series)
        assert isinstance(report.cjv, float)

    def test_low_snr_threshold_fails(self):
        """Force SNR fail by setting threshold very high."""
        checker = SeriesQCChecker(min_snr=1e9)
        series = make_series()
        report = checker.check(series)
        assert not report.passed
        assert any("LOW_SNR" in i for i in report.issues)

    def test_series_uid_preserved_in_report(self):
        checker = SeriesQCChecker()
        series = make_series()
        report = checker.check(series)
        assert report.series_uid == series.series_uid

    def test_modality_preserved_in_report(self):
        checker = SeriesQCChecker()
        series = make_series()
        report = checker.check(series)
        assert report.modality == "MR"
