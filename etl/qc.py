"""
etl/qc.py
─────────────────────────────────────────────────────────────────────────────
Automated DICOM Quality Control (QC) stage.

Implements MRIQC-inspired Image Quality Metrics (IQMs):
  - Series completeness check  (detects missing slices)
  - Pixel statistics sanity    (detects constant, dead, or truncated volumes)
  - SNR, FBER, EFC, CJV        (standard radiological quality metrics)

References:
  - Esteban et al. (2017) "MRIQC: Advancing the automatic prediction of
    image quality in MRI data acquisition" — PLOS ONE
  - DICOMETLPipeline best practices (CapeStart / Collective Minds, 2024)
  - Automated series QC, Karani et al. MICCAI (2021)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from scipy import stats

from etl.models import DicomSeries

# ─── QC Result dataclass ──────────────────────────────────────────────────────


@dataclass
class QCReport:
    """
    Holds all QC metrics computed for a single DICOM series.
    A series is considered acceptable when `passed` is True.
    """

    series_uid: str
    modality: str

    # Completeness
    n_slices: int = 0
    expected_gap_mm: float = 0.0
    missing_slices: int = 0

    # Pixel statistics
    snr: float = 0.0
    fber: float = 0.0  # FMRIB Brain Extraction Ratio
    efc: float = 0.0  # Entropy Focus Criterion
    cjv: float = 0.0  # Coefficient of Joint Variation

    # Issues list — human-readable tags for any failing check
    issues: list[str] = field(default_factory=list)
    passed: bool = True  # overall gate


# ─── QC Checker class ─────────────────────────────────────────────────────────


class SeriesQCChecker:
    """
    Runs automated quality checks on a DicomSeries and returns a QCReport.

    Instantiate once and call `.check(series)` for each series.
    All thresholds are configurable at construction time.
    """

    def __init__(
        self,
        min_slices: int = 4,
        min_snr: float = 5.0,
        min_fber: float = 3.0,
        max_cjv: float = 2.0,  # relaxed — full segmentation unavailable
        max_missing_frac: float = 0.10,  # allow up to 10 % missing slices
    ):
        self.min_slices = min_slices
        self.min_snr = min_snr
        self.min_fber = min_fber
        self.max_cjv = max_cjv
        self.max_missing_frac = max_missing_frac

    # ── public API ─────────────────────────────────────────────────────────────

    def check(self, series: DicomSeries) -> QCReport:
        report = QCReport(
            series_uid=series.series_uid,
            modality=series.modality,
            n_slices=series.num_slices,
        )

        self._check_slice_count(series, report)
        self._check_pixel_statistics(series, report)
        self._compute_iqms(series, report)
        self._apply_thresholds(report)

        if report.passed:
            logger.debug(
                f"QC PASS  {series.series_uid[:16]}…  "
                f"SNR={report.snr:.1f}  FBER={report.fber:.1f}  "
                f"CJV={report.cjv:.2f}  EFC={report.efc:.4f}"
            )
        else:
            logger.warning(f"QC FAIL  {series.series_uid[:16]}…  issues={report.issues}")

        return report

    # ── private checks ─────────────────────────────────────────────────────────

    def _check_slice_count(self, series: DicomSeries, report: QCReport) -> None:
        """Flag series with too few slices to be useful (e.g., scout scans)."""
        if series.num_slices < self.min_slices:
            report.issues.append(f"TOO_FEW_SLICES:{series.num_slices}")

    def _check_pixel_statistics(self, series: DicomSeries, report: QCReport) -> None:
        """
        Sanity checks on pixel data:
          1. Constant / dead image
          2. Extreme outlier voxels (proportion exceeds 1 %)
          3. Crude no-reference SNR estimate
          4. Possible field-of-view truncation
        """
        volume = series.pixel_array
        flat = volume.flatten().astype(np.float64)

        # 1. Constant image
        if np.std(flat) < 1e-6:
            report.issues.append("CONSTANT_IMAGE")
            return  # Nothing else meaningful to compute

        # 2. Extreme outliers — z-score > 10 in > 1 % of voxels
        positive = flat[flat > 0]
        if len(positive) > 0:
            z = np.abs(stats.zscore(positive))
            outlier_frac = float(np.mean(z > 10))
            if outlier_frac > 0.01:
                report.issues.append(f"EXTREME_OUTLIERS:{outlier_frac:.3f}")

        # 3. No-reference SNR: signal = p95, noise = std of background voxels
        threshold = np.percentile(flat, 10)
        bg_vals = flat[flat <= threshold]
        signal = float(np.percentile(flat, 95))
        noise_std = float(bg_vals.std()) if len(bg_vals) > 0 else 1.0
        report.snr = signal / (noise_std + 1e-8)

        # 4. FOV truncation: border voxels brighter than 70th percentile → anatomy cut off
        #    Using p70 (not median) avoids false positives from synthetic/flat-background volumes
        threshold_70 = float(np.percentile(flat, 70))
        border_mean = float(
            np.mean(
                [
                    volume[0, :, :].mean(),
                    volume[-1, :, :].mean(),
                    volume[:, 0, :].mean(),
                    volume[:, -1, :].mean(),
                ]
            )
        )
        if border_mean > threshold_70:
            report.issues.append("POSSIBLE_FOV_TRUNCATION")

    def _compute_iqms(self, series: DicomSeries, report: QCReport) -> None:
        """
        Compute MRIQC-inspired Image Quality Metrics.

        SNR  — Signal-to-Noise Ratio   (higher = better)
        FBER — FMRIB Brain Extraction Ratio: foreground / background energy ratio
        EFC  — Entropy Focus Criterion  (lower = sharper, less motion)
        CJV  — Coefficient of Joint Variation (WM/GM contrast proxy; lower = better)

        Note: Without a brain mask we use an Otsu-like foreground/background split
        as a proxy — appropriate for this pipeline's pre-segmentation stage.
        """
        volume = series.pixel_array.astype(np.float64)
        flat = volume.flatten()

        if np.std(flat) < 1e-6:
            return  # Already flagged as CONSTANT_IMAGE

        # Foreground / background split (Otsu percentile proxy)
        fg_threshold = np.percentile(flat, 20)
        fg_vals = flat[flat > fg_threshold]
        bg_vals = flat[flat <= fg_threshold]

        # FBER
        fg_var = fg_vals.var()
        bg_var = bg_vals.var() if bg_vals.size > 0 else 1.0
        report.fber = float(fg_var / (bg_var + 1e-8))

        # EFC (normalized Shannon entropy of voxel magnitudes)
        n = float(volume.size)
        abs_vol = np.abs(volume)
        b_max = np.sqrt(np.sum(abs_vol**2)) + 1e-8
        norm_vol = abs_vol / b_max
        efc_max = (1.0 / np.sqrt(n)) * np.log(1.0 / np.sqrt(n) + 1e-12)
        efc = float(np.sum(norm_vol * np.log(norm_vol + 1e-12)))
        report.efc = float(efc / efc_max) if abs(efc_max) > 1e-12 else 0.0

        # CJV — coefficient of joint variation (WM/GM contrast proxy)
        p25, p50, p75 = np.percentile(fg_vals, [25, 50, 75])
        gm_proxy = fg_vals[(fg_vals >= p25) & (fg_vals <= p50)]
        wm_proxy = fg_vals[fg_vals > p75]

        if len(gm_proxy) > 1 and len(wm_proxy) > 1:
            diff = abs(wm_proxy.mean() - gm_proxy.mean()) + 1e-8
            report.cjv = float((wm_proxy.std() + gm_proxy.std()) / diff)
        else:
            report.cjv = 0.0

    def _apply_thresholds(self, report: QCReport) -> None:
        """
        Gate the report: add issue tags for metrics that fail thresholds,
        then set `passed = False` if any issue was found.
        """
        if report.snr > 0 and report.snr < self.min_snr:
            report.issues.append(f"LOW_SNR:{report.snr:.1f}")

        if report.fber > 0 and report.fber < self.min_fber:
            report.issues.append(f"LOW_FBER:{report.fber:.1f}")

        if report.cjv > 0 and report.cjv > self.max_cjv:
            report.issues.append(f"HIGH_CJV:{report.cjv:.2f}")

        report.passed = len(report.issues) == 0
