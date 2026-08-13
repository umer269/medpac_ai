"""
etl/loader.py
─────────────
LOAD stage — saves the transformed series as a NIfTI file, writes a PNG
preview, and upserts the study metadata into the SQLite database.
"""

from __future__ import annotations

import datetime
import os
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
from loguru import logger
from PIL import Image

from etl.models import DicomStudyRecord, TransformedSeries

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class DicomLoader:
    """
    Persists a TransformedSeries to disk and records metadata in the DB.
    """

    def __init__(
        self,
        nifti_dir: str,
        preview_dir: str,
        db_session: Session,
        save_preview: bool = True,
    ):
        self.nifti_dir = nifti_dir
        self.preview_dir = preview_dir
        self.db_session = db_session
        self.save_preview = save_preview
        os.makedirs(nifti_dir, exist_ok=True)
        os.makedirs(preview_dir, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────────

    def load(self, transformed: TransformedSeries) -> DicomStudyRecord:
        """
        Save NIfTI + preview, write DB record, return the persisted record.
        Raises on critical failures so the pipeline can mark the study as
        failed without crashing.
        """
        source = transformed.source
        logger.info(
            f"[Load] Saving series {source.series_uid[:16]}… "
            f"→ NIfTI shape={transformed.normalized_array.shape}"
        )

        # ── Existing record? Update it; otherwise insert a new one ────────────
        record = (
            self.db_session.query(DicomStudyRecord).filter_by(series_uid=source.series_uid).first()
        )
        if record is None:
            record = DicomStudyRecord(
                study_uid=source.study_uid,
                series_uid=source.series_uid,
                patient_id=source.patient_id,
                patient_name=source.patient_name,
                modality=source.modality,
                study_date=source.study_date,
                study_desc=source.study_desc,
                num_slices=source.num_slices,
                rows=source.rows,
                cols=source.cols,
                pixel_spacing_x=source.pixel_spacing[0],
                pixel_spacing_y=source.pixel_spacing[1],
                slice_thickness=source.slice_thickness,
            )
            self.db_session.add(record)

        try:
            # 1. Save NIfTI ────────────────────────────────────────────────────
            nifti_path = self._save_nifti(transformed)
            record.nifti_path = nifti_path

            # 2. Save Segmentation NIfTI if present ─────────────────────────────
            if transformed.segmentation_array is not None:
                seg_path = self._save_segmentation(transformed)
                record.segmentation_path = seg_path
                record.segmentation_volume_cc = transformed.segmentation_volume_cc
            else:
                record.segmentation_path = None
                record.segmentation_volume_cc = None

            # 3. Save preview PNG ──────────────────────────────────────────────
            if self.save_preview and transformed.preview_slice is not None:
                preview_path = self._save_preview(transformed)
                record.preview_path = preview_path

            # 4. Mark success ──────────────────────────────────────────────────
            record.status = "success"
            record.error_message = None
            record.processed_at = datetime.datetime.now(datetime.UTC)
            self.db_session.commit()

            logger.success(f"Series {source.series_uid[:16]}… saved (NIfTI → {nifti_path})")

        except Exception as exc:
            self.db_session.rollback()
            record.status = "failed"
            record.error_message = str(exc)
            record.processed_at = datetime.datetime.now(datetime.UTC)
            self.db_session.add(record)
            self.db_session.commit()
            logger.error(f"Failed to load series {source.series_uid[:16]}…: {exc}")
            raise

        return record

    # ── private helpers ───────────────────────────────────────────────────────

    def _save_nifti(self, transformed: TransformedSeries) -> str:
        """
        Save the normalized 3-D array as a compressed NIfTI (.nii.gz) file.
        NIfTI convention: axes must be (X, Y, Z), so we transpose from (Z, Y, X).
        """
        volume = transformed.normalized_array.astype(np.float32)
        # DICOM stores (slices, rows, cols) = (Z, Y, X); NIfTI wants (X, Y, Z)
        volume_nifti = np.transpose(volume, (2, 1, 0))

        nii_img = nib.Nifti1Image(volume_nifti, affine=transformed.affine)
        nii_img.header.set_xyzt_units(xyz="mm")

        safe_uid = transformed.source.series_uid.replace(".", "_")
        filename = f"{safe_uid}.nii.gz"
        filepath = os.path.join(self.nifti_dir, filename)
        nib.save(nii_img, filepath)
        return filepath

    def _save_segmentation(self, transformed: TransformedSeries) -> str:
        """Save the binary 3-D segmentation mask as a compressed NIfTI file."""
        mask = transformed.segmentation_array.astype(np.uint8)
        mask_nifti = np.transpose(mask, (2, 1, 0))

        nii_img = nib.Nifti1Image(mask_nifti, affine=transformed.affine)
        nii_img.header.set_xyzt_units(xyz="mm")

        safe_uid = transformed.source.series_uid.replace(".", "_")
        filename = f"{safe_uid}_seg.nii.gz"
        filepath = os.path.join(self.nifti_dir, filename)
        nib.save(nii_img, filepath)
        return filepath

    def _save_preview(self, transformed: TransformedSeries) -> str:
        """Save the 2-D preview slice (uint8) as a PNG file, with color overlay if segmentation is present."""
        safe_uid = transformed.source.series_uid.replace(".", "_")
        filename = f"{safe_uid}_preview.png"
        filepath = os.path.join(self.preview_dir, filename)

        gray = transformed.preview_slice
        if transformed.segmentation_array is not None:
            # Extract corresponding slice of the segmentation mask
            axis = transformed.preview_slice_axis
            idx = transformed.segmentation_array.shape[axis] // 2
            if axis == 0:
                seg_slice = transformed.segmentation_array[idx, :, :]
            elif axis == 1:
                seg_slice = transformed.segmentation_array[:, idx, :]
            else:
                seg_slice = transformed.segmentation_array[:, :, idx]

            # Convert gray to RGB
            rgb = np.stack([gray, gray, gray], axis=-1)

            # Apply overlay where mask is positive
            mask = seg_slice > 0
            if np.any(mask):
                # Choose color based on modality: MR = red/orange, CT = cyan
                if transformed.source.modality == "MR":
                    color = np.array(
                        [255, 80, 80], dtype=np.uint8
                    )  # semi-transparent reddish-orange
                else:
                    color = np.array([80, 255, 255], dtype=np.uint8)  # semi-transparent cyan

                alpha = 0.45
                rgb[mask] = (rgb[mask] * (1.0 - alpha) + color * alpha).astype(np.uint8)

            img = Image.fromarray(rgb, mode="RGB")
        else:
            img = Image.fromarray(gray, mode="L")

        img.save(filepath)
        return filepath
