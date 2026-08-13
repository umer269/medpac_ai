"""
etl/extractor.py
─────────────────
EXTRACT stage — scans the DICOM inbox, groups files by SeriesInstanceUID,
and builds a DicomSeries dataclass per series.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pydicom
from loguru import logger
from tqdm import tqdm

from etl.models import DicomSeries

# ─── helpers ─────────────────────────────────────────────────────────────────


def _tag(ds: pydicom.Dataset, keyword: str, default: str = "") -> str:
    """Safely read a DICOM tag, returning a default if it is absent."""
    val = getattr(ds, keyword, None)
    return str(val) if val is not None else default


def _collect_dicom_files(root: str, recursive: bool) -> list[Path]:
    """Return all files under root that appear to be DICOM (by magic bytes)."""
    files: list[Path] = []
    walker = Path(root).rglob("*") if recursive else Path(root).glob("*")
    for path in walker:
        if not path.is_file():
            continue
        # Quick check: DICOM files have 'DICM' at byte offset 128
        try:
            with open(path, "rb") as fh:
                fh.seek(128)
                magic = fh.read(4)
            if magic == b"DICM":
                files.append(path)
        except OSError:
            pass
    return files


def _build_affine(ds: pydicom.Dataset) -> np.ndarray:
    """
    Construct a basic NIfTI affine matrix from DICOM image position /
    orientation tags. Falls back to an identity matrix if tags are absent.
    """
    try:
        iop = ds.ImageOrientationPatient  # [F11, F12, F13, F21, F22, F23]
        ipp = ds.ImagePositionPatient  # [x0, y0, z0]
        ps = ds.PixelSpacing  # [row_spacing, col_spacing]
        st = float(getattr(ds, "SliceThickness", 1.0))

        f = np.array(iop, dtype=float).reshape(2, 3)
        dr = float(ps[0])
        dc = float(ps[1])

        # Row direction cosines × row spacing
        r = f[0] * dc
        # Column direction cosines × column spacing
        c = f[1] * dr
        # Normal (slice direction) — cross product
        k = np.cross(f[0], f[1]) * st

        origin = np.array(ipp, dtype=float)

        affine = np.eye(4)
        affine[:3, 0] = r
        affine[:3, 1] = c
        affine[:3, 2] = k
        affine[:3, 3] = origin
        return affine
    except Exception:
        return np.eye(4)


# ─── main Extractor class ─────────────────────────────────────────────────────


class DicomExtractor:
    """
    Scans the source directory for DICOM files, groups them by series,
    and yields fully-loaded DicomSeries objects.
    """

    def __init__(self, source_dir: str, modalities: list[str], recursive: bool = True):
        self.source_dir = source_dir
        self.modalities = [m.upper() for m in modalities]
        self.recursive = recursive

    # ── public API ────────────────────────────────────────────────────────────

    def extract(self) -> Generator[DicomSeries, None, None]:
        """
        Yields one DicomSeries per discovered DICOM series in the inbox.
        Files not matching the configured modalities are silently skipped.
        """
        if not os.path.isdir(self.source_dir):
            logger.warning(
                f"Source directory not found: {self.source_dir!r}. No files will be extracted."
            )
            return

        logger.info(f"Scanning '{self.source_dir}' for DICOM files …")
        dicom_files = _collect_dicom_files(self.source_dir, self.recursive)
        logger.info(f"Found {len(dicom_files)} DICOM file(s). Grouping by series …")

        # Group file paths by SeriesInstanceUID
        series_map: dict[str, list[pydicom.Dataset]] = defaultdict(list)

        for path in tqdm(dicom_files, desc="[Extract] Reading DICOM headers"):
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=True)
                series_uid = _tag(ds, "SeriesInstanceUID") or str(path)
                # Filter by modality
                modality = _tag(ds, "Modality").upper()
                if modality not in self.modalities:
                    logger.debug(f"Skipping {path.name}: modality '{modality}' not in config.")
                    continue
                series_map[series_uid].append(ds)
            except Exception as exc:
                logger.warning(f"Cannot read {path.name}: {exc}")

        logger.info(f"Grouped into {len(series_map)} series.")

        for series_uid, header_list in series_map.items():
            series = self._load_series(series_uid, header_list)
            if series is not None:
                yield series

    # ── private helpers ───────────────────────────────────────────────────────

    def _load_series(
        self,
        series_uid: str,
        header_list: list[pydicom.Dataset],
    ) -> DicomSeries | None:
        """
        Given a list of DICOM dataset headers (same series), re-read each file
        WITH pixel data, sort by ImagePositionPatient z-coordinate, and stack
        into a 3-D numpy array (slices × rows × cols).
        """
        try:
            # Re-read with pixel data
            full_datasets = []
            for ds in header_list:
                full_ds = pydicom.dcmread(ds.filename)
                full_datasets.append(full_ds)

            # Sort slices by z-position (or InstanceNumber as fallback)
            def sort_key(d: pydicom.Dataset) -> float:
                try:
                    return float(d.ImagePositionPatient[2])
                except Exception:
                    return float(getattr(d, "InstanceNumber", 0))

            full_datasets.sort(key=sort_key)

            # Stack pixel arrays
            slices = []
            for d in full_datasets:
                arr = d.pixel_array.astype(np.float32)
                # Apply DICOM rescale slope/intercept (Hounsfield for CT)
                slope = float(getattr(d, "RescaleSlope", 1.0))
                intercept = float(getattr(d, "RescaleIntercept", 0.0))
                arr = arr * slope + intercept
                slices.append(arr)

            volume = np.stack(slices, axis=0)  # (S, R, C)

            ref = full_datasets[0]
            ps = getattr(ref, "PixelSpacing", [1.0, 1.0])

            return DicomSeries(
                series_uid=series_uid,
                study_uid=_tag(ref, "StudyInstanceUID"),
                patient_id=_tag(ref, "PatientID"),
                patient_name=_tag(ref, "PatientName"),
                modality=_tag(ref, "Modality"),
                study_date=_tag(ref, "StudyDate"),
                study_desc=_tag(ref, "StudyDescription"),
                num_slices=len(full_datasets),
                rows=int(ref.Rows),
                cols=int(ref.Columns),
                pixel_spacing=(float(ps[0]), float(ps[1])),
                slice_thickness=float(getattr(ref, "SliceThickness", 1.0)),
                pixel_array=volume,
            )
        except Exception as exc:
            logger.error(f"Failed to load series '{series_uid}': {exc}")
            return None
