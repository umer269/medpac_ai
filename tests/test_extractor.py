"""
tests/test_extractor.py
────────────────────────
Unit tests for the EXTRACT stage (DICOM parsing and series grouping).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pydicom

from etl.extractor import DicomExtractor, _build_affine, _collect_dicom_files, _tag
from etl.models import DicomSeries


def test_tag_helper():
    # Test case where tag exists
    ds = pydicom.Dataset()
    ds.PatientName = "Doe^John"
    assert _tag(ds, "PatientName") == "Doe^John"

    # Test case where tag is missing
    assert _tag(ds, "Modality", default="MR") == "MR"


def test_build_affine_fallback():
    # Empty dataset should fallback to identity matrix
    ds = pydicom.Dataset()
    affine = _build_affine(ds)
    assert np.allclose(affine, np.eye(4))


def test_build_affine_success():
    ds = pydicom.Dataset()
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.ImagePositionPatient = [10.0, 20.0, 30.0]
    ds.PixelSpacing = [1.5, 1.5]
    ds.SliceThickness = 3.0

    affine = _build_affine(ds)
    assert affine.shape == (4, 4)
    # Check rotation scale values
    assert affine[0, 0] == 1.5
    assert affine[1, 1] == 1.5
    assert affine[2, 2] == 3.0
    # Check origin
    assert np.allclose(affine[:3, 3], [10.0, 20.0, 30.0])


def test_collect_dicom_files(tmp_path: Path):
    # Create a file with the DICOM magic bytes 'DICM' at offset 128
    dicom_file = tmp_path / "valid.dcm"
    with open(dicom_file, "wb") as f:
        f.write(b"\x00" * 128)
        f.write(b"DICM")
        f.write(b"some other content")

    # Create a file without the magic bytes
    non_dicom_file = tmp_path / "invalid.txt"
    with open(non_dicom_file, "wb") as f:
        f.write(b"Hello World" * 20)

    files = _collect_dicom_files(str(tmp_path), recursive=False)
    assert len(files) == 1
    assert files[0] == dicom_file


@patch("etl.extractor._collect_dicom_files")
@patch("pydicom.dcmread")
def test_dicom_extractor_run(mock_dcmread, mock_collect, tmp_path: Path):
    mock_file = tmp_path / "slice1.dcm"
    mock_collect.return_value = [mock_file]

    # Mock dataset representing a DICOM file
    ds_header = MagicMock(spec=pydicom.Dataset)
    ds_header.filename = str(mock_file)
    ds_header.SeriesInstanceUID = "1.2.3"
    ds_header.StudyInstanceUID = "1.2"
    ds_header.Modality = "MR"
    ds_header.PatientID = "P123"
    ds_header.PatientName = "Umer^Raja"
    ds_header.StudyDate = "20260813"
    ds_header.StudyDescription = "Brain MRI"
    ds_header.Rows = 64
    ds_header.Columns = 64
    ds_header.PixelSpacing = [1.0, 1.0]
    ds_header.SliceThickness = 3.0
    ds_header.ImagePositionPatient = [0.0, 0.0, 10.0]
    ds_header.pixel_array = np.ones((64, 64), dtype=np.uint16)
    ds_header.RescaleSlope = 1.0
    ds_header.RescaleIntercept = 0.0

    # Ensure dcmread returns the mock dataset
    mock_dcmread.return_value = ds_header

    extractor = DicomExtractor(source_dir=str(tmp_path), modalities=["MR"], recursive=False)
    series_list = list(extractor.extract())

    assert len(series_list) == 1
    series = series_list[0]
    assert isinstance(series, DicomSeries)
    assert series.series_uid == "1.2.3"
    assert series.modality == "MR"
    assert series.num_slices == 1
    assert series.rows == 64
    assert series.cols == 64
    assert series.pixel_array.shape == (1, 64, 64)
