"""
etl/models.py
─────────────
SQLAlchemy ORM models + dataclasses used throughout the pipeline.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sqlalchemy import (
    Column, DateTime, Float, Integer, String, create_engine
)
from sqlalchemy.orm import DeclarativeBase, Session


# ─── SQLAlchemy ORM ──────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class DicomStudyRecord(Base):
    """Persisted record for every DICOM series processed by the pipeline."""

    __tablename__ = "dicom_studies"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    study_uid       = Column(String, nullable=False, index=True)
    series_uid      = Column(String, nullable=False, unique=True, index=True)
    patient_id      = Column(String, nullable=True)
    patient_name    = Column(String, nullable=True)
    modality        = Column(String, nullable=False)
    study_date      = Column(String, nullable=True)
    study_desc      = Column(String, nullable=True)
    num_slices      = Column(Integer, nullable=True)
    rows            = Column(Integer, nullable=True)
    cols            = Column(Integer, nullable=True)
    pixel_spacing_x         = Column(Float, nullable=True)
    pixel_spacing_y         = Column(Float, nullable=True)
    slice_thickness         = Column(Float, nullable=True)
    nifti_path              = Column(String, nullable=True)
    segmentation_path       = Column(String, nullable=True)
    segmentation_volume_cc  = Column(Float, nullable=True)
    preview_path            = Column(String, nullable=True)
    status                  = Column(String, nullable=False, default="pending")   # pending | success | failed
    error_message           = Column(String, nullable=True)
    processed_at            = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"<DicomStudyRecord series={self.series_uid!r} "
            f"modality={self.modality!r} status={self.status!r}>"
        )


def init_db(db_path: str) -> Session:
    """Create (or connect to) the SQLite metadata database and return a session."""
    engine = create_engine(f"sqlite:///{db_path}", echo=False, future=True)
    Base.metadata.create_all(engine)
    return Session(engine)


# ─── In-memory dataclasses (passed between pipeline stages) ──────────────────

@dataclass
class DicomSeries:
    """
    Represents a fully-loaded DICOM series extracted from the inbox.
    Holds the raw pixel volume and key DICOM metadata tags.
    """

    series_uid:      str
    study_uid:       str
    patient_id:      str
    patient_name:    str
    modality:        str
    study_date:      str
    study_desc:      str
    num_slices:      int
    rows:            int
    cols:            int
    pixel_spacing:   tuple[float, float]    # (row_spacing_mm, col_spacing_mm)
    slice_thickness: float
    # Shape: (slices, rows, cols) — raw Hounsfield / signal values
    pixel_array:     np.ndarray = field(repr=False)


@dataclass
class TransformedSeries:
    """
    A normalized, resampled volume ready to be saved as NIfTI.
    """

    source: DicomSeries                    # reference to the original
    normalized_array: np.ndarray = field(repr=False)
    affine: np.ndarray            = field(repr=False)   # 4×4 NIfTI affine matrix
    preview_slice: Optional[np.ndarray] = field(default=None, repr=False)
    segmentation_array: Optional[np.ndarray] = field(default=None, repr=False)
    preview_slice_axis: int = field(default=0, repr=False)
    segmentation_volume_cc: Optional[float] = field(default=None, repr=False)


