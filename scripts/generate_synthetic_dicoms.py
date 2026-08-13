"""
scripts/generate_synthetic_dicoms.py
──────────────────────────────────────
Generates a small synthetic DICOM dataset for pipeline testing.

Creates 2 series:
  - Series A: Synthetic T1-weighted brain MRI (30 slices, 128×128)
  - Series B: Synthetic CT abdomen (25 slices, 128×128) — intentionally
              fewer slices to also test the QC slice-count check

Run from the project root:
    python scripts/generate_synthetic_dicoms.py
"""

import os

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    generate_uid,
)

# ─── helpers ──────────────────────────────────────────────────────────────────

def make_file_meta(sop_class_uid: str, sop_instance_uid: str) -> FileMetaDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID    = sop_class_uid
    meta.MediaStorageSOPInstanceUID = sop_instance_uid
    meta.TransferSyntaxUID          = ExplicitVRLittleEndian
    meta.FileMetaInformationVersion = b"\x00\x01"
    return meta


def write_slice(
    out_path: str,
    pixel_array_2d: np.ndarray,
    series_uid:     str,
    study_uid:      str,
    sop_uid:        str,
    patient_id:     str,
    patient_name:   str,
    modality:       str,
    study_desc:     str,
    series_desc:    str,
    slice_index:    int,
    z_position:     float,
    pixel_spacing:  tuple = (1.0, 1.0),
    slice_thickness: float = 3.0,
    study_date:     str = "20240601",
) -> None:
    """Write one DICOM slice to disk."""

    # SOP class UIDs
    sop_class = {
        "MR": "1.2.840.10008.5.1.4.1.1.4",   # MR Image Storage
        "CT": "1.2.840.10008.5.1.4.1.1.2",   # CT Image Storage
    }.get(modality, "1.2.840.10008.5.1.4.1.1.4")

    meta = make_file_meta(sop_class, sop_uid)
    ds   = FileDataset(out_path, {}, file_meta=meta, preamble=b"\x00" * 128)

    # ── Patient / Study / Series ──────────────────────────────────────────────
    ds.PatientName        = patient_name
    ds.PatientID          = patient_id
    ds.StudyInstanceUID   = study_uid
    ds.SeriesInstanceUID  = series_uid
    ds.SOPInstanceUID     = sop_uid
    ds.SOPClassUID        = sop_class
    ds.StudyDate          = study_date
    ds.StudyTime          = "120000"
    ds.Modality           = modality
    ds.StudyDescription   = study_desc
    ds.SeriesDescription  = series_desc
    ds.InstanceNumber     = slice_index + 1

    # ── Image geometry ────────────────────────────────────────────────────────
    ds.Rows               = pixel_array_2d.shape[0]
    ds.Columns            = pixel_array_2d.shape[1]
    ds.PixelSpacing       = list(pixel_spacing)
    ds.SliceThickness     = slice_thickness
    ds.ImagePositionPatient      = [0.0, 0.0, z_position]
    ds.ImageOrientationPatient   = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.SliceLocation             = z_position

    # ── Pixel encoding ────────────────────────────────────────────────────────
    ds.SamplesPerPixel    = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated      = 16
    ds.BitsStored         = 16
    ds.HighBit            = 15
    ds.PixelRepresentation = 1     # signed (Hounsfield for CT)
    ds.RescaleSlope       = 1.0
    ds.RescaleIntercept   = 0.0 if modality != "CT" else -1024.0

    arr_int = pixel_array_2d.astype(np.int16)
    ds.PixelData = arr_int.tobytes()

    ds.is_implicit_VR  = False
    ds.is_little_endian = True

    pydicom.dcmwrite(out_path, ds)


# ─── Synthetic brain MRI (T1w) ────────────────────────────────────────────────

def generate_brain_mri(out_dir: str, n_slices: int = 30, size: int = 128) -> None:
    """
    T1-weighted MRI: dark background, CSF (low), GM (mid), WM (high) rings.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng        = np.random.default_rng(42)
    study_uid  = generate_uid()
    series_uid = generate_uid()

    cx, cy = size // 2, size // 2
    Y, X   = np.ogrid[:size, :size]

    for i in range(n_slices):
        # Elliptical skull profile varies slightly by slice
        radius = int(size * 0.38 * (1 - 0.3 * abs(i / n_slices - 0.5)))
        dist   = np.sqrt(((X - cx) / (radius + 1)) ** 2 + ((Y - cy) / (radius + 1)) ** 2)

        arr = np.zeros((size, size), dtype=np.float32)
        arr[dist < 1.0] = 400  + rng.normal(0, 30, (size, size))[dist < 1.0]  # GM
        arr[dist < 0.7] = 700  + rng.normal(0, 35, (size, size))[dist < 0.7]  # WM
        arr[dist < 0.3] = 100  + rng.normal(0, 20, (size, size))[dist < 0.3]  # CSF
        arr = np.clip(arr, 0, 4095)

        sop_uid = generate_uid()
        fname   = os.path.join(out_dir, f"brain_mri_{i:03d}.dcm")
        write_slice(
            out_path        = fname,
            pixel_array_2d  = arr.astype(np.int16),
            series_uid      = series_uid,
            study_uid       = study_uid,
            sop_uid         = sop_uid,
            patient_id      = "SYNTH001",
            patient_name    = "Synthetic^BrainMRI",
            modality        = "MR",
            study_desc      = "Brain MRI Synthetic",
            series_desc     = "T1w Axial",
            slice_index     = i,
            z_position      = float(i * 3.0),
            pixel_spacing   = (1.0, 1.0),
            slice_thickness = 3.0,
        )

    print(f"  Brain MRI series: {n_slices} slices -> {out_dir}")


# ─── Synthetic CT abdomen ─────────────────────────────────────────────────────

def generate_ct_abdomen(out_dir: str, n_slices: int = 25, size: int = 128) -> None:
    """
    CT abdomen (HU units): air background, soft tissue core, bone ring.
    RescaleIntercept = -1024 → raw values shifted to HU range.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng        = np.random.default_rng(7)
    study_uid  = generate_uid()
    series_uid = generate_uid()

    cx, cy = size // 2, size // 2
    Y, X   = np.ogrid[:size, :size]

    for i in range(n_slices):
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        # In raw pixel values (HU = pixel + intercept = pixel - 1024)
        # Air  : HU ~ -1000 → raw ~ 24
        # Fat  : HU ~  -100 → raw ~ 924
        # Soft : HU ~    50 → raw ~ 1074
        # Bone : HU ~   700 → raw ~ 1724

        arr = np.full((size, size), 24, dtype=np.float32)      # air
        arr[dist < size * 0.45] = 924  + rng.normal(0, 20, (size, size))[dist < size * 0.45]  # fat
        arr[dist < size * 0.35] = 1074 + rng.normal(0, 30, (size, size))[dist < size * 0.35]  # soft tissue
        arr[dist < size * 0.42][dist[dist < size * 0.42] > size * 0.38] = 1724  # bone ring
        arr = np.clip(arr, 0, 4095)

        sop_uid = generate_uid()
        fname   = os.path.join(out_dir, f"ct_abdomen_{i:03d}.dcm")
        write_slice(
            out_path        = fname,
            pixel_array_2d  = arr.astype(np.int16),
            series_uid      = series_uid,
            study_uid       = study_uid,
            sop_uid         = sop_uid,
            patient_id      = "SYNTH002",
            patient_name    = "Synthetic^CTAbdomen",
            modality        = "CT",
            study_desc      = "CT Abdomen Synthetic",
            series_desc     = "Axial 5mm",
            slice_index     = i,
            z_position      = float(i * 5.0),
            pixel_spacing   = (0.75, 0.75),
            slice_thickness = 5.0,
        )

    print(f"  CT Abdomen series: {n_slices} slices -> {out_dir}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    inbox = os.path.join("data", "dicom_inbox")
    print("\nGenerating synthetic DICOM dataset ...\n")

    generate_brain_mri(os.path.join(inbox, "brain_mri"))
    generate_ct_abdomen(os.path.join(inbox, "ct_abdomen"))

    total = sum(
        len(os.listdir(os.path.join(inbox, d)))
        for d in os.listdir(inbox)
        if os.path.isdir(os.path.join(inbox, d))
    )
    print(f"\nDone. {total} DICOM files written to '{inbox}/'")
    print("   Run the pipeline:  python main.py\n")
