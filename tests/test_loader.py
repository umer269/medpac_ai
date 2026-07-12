"""
tests/test_loader.py
─────────────────────
Unit tests for the Load stage.
Uses a temporary SQLite database and in-memory data — no real DICOM files needed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from etl.loader  import DicomLoader
from etl.models  import Base, DicomSeries, TransformedSeries


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dirs(tmp_path):
    nifti_dir   = tmp_path / "nifti"
    preview_dir = tmp_path / "previews"
    nifti_dir.mkdir()
    preview_dir.mkdir()
    return str(nifti_dir), str(preview_dir)


@pytest.fixture
def db_session(tmp_path):
    """Create a fresh SQLite engine + session; dispose on teardown."""
    db_path = str(tmp_path / "test_metadata.db")
    engine  = create_engine(f"sqlite:///{db_path}", echo=False, future=True)
    Base.metadata.create_all(engine)
    session = Session(engine)
    yield session
    session.close()
    engine.dispose()


def make_transformed(series_uid: str = "1.2.3.4.5") -> TransformedSeries:
    src = DicomSeries(
        series_uid      = series_uid,
        study_uid       = "1.2.3",
        patient_id      = "P001",
        patient_name    = "Doe^John",
        modality        = "MR",
        study_date      = "20240601",
        study_desc      = "Brain MRI",
        num_slices      = 10,
        rows            = 32,
        cols            = 32,
        pixel_spacing   = (1.0, 1.0),
        slice_thickness = 3.0,
        pixel_array     = np.zeros((10, 32, 32), dtype=np.float32),
    )
    normalized = np.random.rand(10, 32, 32).astype(np.float32)
    preview    = (np.random.rand(32, 32) * 255).astype(np.uint8)
    affine     = np.eye(4, dtype=float)
    return TransformedSeries(
        source           = src,
        normalized_array = normalized,
        affine           = affine,
        preview_slice    = preview,
    )


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestDicomLoader:

    def test_nifti_file_created(self, tmp_dirs, db_session):
        nifti_dir, preview_dir = tmp_dirs
        loader  = DicomLoader(nifti_dir, preview_dir, db_session, save_preview=False)
        record  = loader.load(make_transformed())
        assert record.nifti_path is not None
        assert os.path.isfile(record.nifti_path)

    def test_preview_png_created(self, tmp_dirs, db_session):
        nifti_dir, preview_dir = tmp_dirs
        loader  = DicomLoader(nifti_dir, preview_dir, db_session, save_preview=True)
        record  = loader.load(make_transformed())
        assert record.preview_path is not None
        assert os.path.isfile(record.preview_path)
        assert record.preview_path.endswith(".png")

    def test_db_record_status_success(self, tmp_dirs, db_session):
        nifti_dir, preview_dir = tmp_dirs
        loader  = DicomLoader(nifti_dir, preview_dir, db_session, save_preview=False)
        record  = loader.load(make_transformed())
        assert record.status == "success"

    def test_duplicate_series_is_upserted(self, tmp_dirs, db_session):
        """Loading the same series twice should not create duplicate DB rows."""
        from etl.models import DicomStudyRecord
        nifti_dir, preview_dir = tmp_dirs
        loader  = DicomLoader(nifti_dir, preview_dir, db_session, save_preview=False)
        series_uid = "9.8.7.6.5"
        loader.load(make_transformed(series_uid))
        loader.load(make_transformed(series_uid))
        count = db_session.query(DicomStudyRecord).filter_by(series_uid=series_uid).count()
        assert count == 1

    def test_metadata_fields_persisted(self, tmp_dirs, db_session):
        nifti_dir, preview_dir = tmp_dirs
        loader  = DicomLoader(nifti_dir, preview_dir, db_session, save_preview=False)
        record  = loader.load(make_transformed("1.2.3.4.5"))
        assert record.patient_id   == "P001"
        assert record.modality     == "MR"
        assert record.num_slices   == 10
        assert record.rows         == 32
        assert record.cols         == 32
