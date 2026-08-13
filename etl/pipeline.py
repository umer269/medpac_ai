"""
etl/pipeline.py
───────────────
Orchestrates the Extract → Transform → Load stages.
Reads config.yaml, wires up the three stages, and reports a summary.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from etl.extractor import DicomExtractor
from etl.loader import DicomLoader
from etl.models import init_db
from etl.qc import SeriesQCChecker
from etl.segmenter import DicomSegmenter
from etl.transformer import DicomTransformer

# ─── config helpers ───────────────────────────────────────────────────────────


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
        ),
        colorize=True,
    )
    # Also write structured logs to file
    log_file = Path("output") / "pipeline.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), level="DEBUG", rotation="10 MB", retention="7 days")


# ─── Pipeline class ───────────────────────────────────────────────────────────


class MedPacsETLPipeline:
    """
    End-to-end orchestrator.

    Usage:
        pipeline = MedPacsETLPipeline("config.yaml")
        pipeline.run()
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        _setup_logging(self.config["pipeline"].get("log_level", "INFO"))

        cfg_ext = self.config["extract"]
        cfg_trn = self.config["transform"]
        cfg_load = self.config["load"]

        # ── Instantiate stages ────────────────────────────────────────────────
        self.extractor = DicomExtractor(
            source_dir=cfg_ext["source_dir"],
            modalities=cfg_ext["modalities"],
            recursive=cfg_ext.get("recursive", True),
        )

        ts = cfg_trn.get("target_spacing")
        self.transformer = DicomTransformer(
            normalization=cfg_trn.get("normalization", "z_score"),
            window_center=float(cfg_trn.get("window_center", 40)),
            window_width=float(cfg_trn.get("window_width", 400)),
            target_spacing=tuple(ts) if ts else None,
            clip_percentile_low=float(cfg_trn.get("clip_percentile_low", 0.5)),
            clip_percentile_high=float(cfg_trn.get("clip_percentile_high", 99.5)),
            save_preview=bool(cfg_trn.get("save_preview", True)),
            preview_slice_axis=int(cfg_trn.get("preview_slice_axis", 2)),
        )

        db_session = init_db(cfg_load["database_path"])
        self.loader = DicomLoader(
            nifti_dir=cfg_load["nifti_dir"],
            preview_dir=cfg_load["preview_dir"],
            db_session=db_session,
            save_preview=bool(cfg_trn.get("save_preview", True)),
        )

        # ── QC checker ───────────────────────────────────────────────────────
        cfg_qc = self.config.get("qc", {})
        self.qc_checker = SeriesQCChecker(
            min_slices=int(cfg_qc.get("min_slices", 4)),
            min_snr=float(cfg_qc.get("min_snr", 5.0)),
            min_fber=float(cfg_qc.get("min_fber", 3.0)),
            max_cjv=float(cfg_qc.get("max_cjv", 2.0)),
            max_missing_frac=float(cfg_qc.get("max_missing_frac", 0.10)),
        )
        self.qc_enabled = bool(cfg_qc.get("enabled", True))

        # ── AI Segmenter ─────────────────────────────────────────────────────
        cfg_seg = self.config.get("segmentation", {})
        self.segmentation_enabled = bool(cfg_seg.get("enabled", True))
        self.segmenter = DicomSegmenter(
            enabled=self.segmentation_enabled,
            model_type=cfg_seg.get("model_type", "mock"),
            model_path=cfg_seg.get("model_path", ""),
            threshold=float(cfg_seg.get("threshold", 0.5)),
            target_structure=cfg_seg.get("target_structure", "ventricles"),
        )

    # ── public API ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the full ETL pipeline and print a summary."""
        logger.info("=" * 60)
        logger.info(
            f"Starting {self.config['pipeline']['name']} v{self.config['pipeline']['version']}"
        )
        logger.info("=" * 60)

        total = success = failed = qc_failed = 0

        for dicom_series in self.extractor.extract():
            total += 1
            try:
                # ── QC gate ───────────────────────────────────────────────────
                if self.qc_enabled:
                    qc_report = self.qc_checker.check(dicom_series)
                    if not qc_report.passed:
                        qc_failed += 1
                        logger.warning(
                            f"Skipping series '{dicom_series.series_uid[:16]}…' "
                            f"— QC failed: {qc_report.issues}"
                        )
                        continue

                transformed = self.transformer.transform(dicom_series)

                # ── AI Segmentation ───────────────────────────────────────────
                if self.segmentation_enabled:
                    effective_spacing = self.transformer.target_spacing or (
                        dicom_series.slice_thickness,
                        dicom_series.pixel_spacing[0],
                        dicom_series.pixel_spacing[1],
                    )
                    mask, vol_cc = self.segmenter.segment(
                        transformed.normalized_array,
                        dicom_series.modality,
                        effective_spacing,
                    )
                    transformed.segmentation_array = mask
                    transformed.segmentation_volume_cc = vol_cc
                    logger.info(
                        f"[Segmentation] Segmented {self.segmenter.target_structure} "
                        f"→ volume: {vol_cc:.2f} cc"
                    )

                self.loader.load(transformed)
                success += 1
            except Exception as exc:
                logger.error(f"Pipeline error for series '{dicom_series.series_uid[:16]}…': {exc}")
                failed += 1

        # ── Summary ───────────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("Pipeline complete.")
        logger.info(f"  Total series processed : {total}")
        logger.info(f"  QC failed (skipped)    : {qc_failed}")
        logger.info(f"  Successful             : {success}")
        logger.info(f"  Failed (errors)        : {failed}")
        logger.info("=" * 60)
