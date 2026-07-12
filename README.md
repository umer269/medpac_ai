# MedPACS-AI

[![CI](https://github.com/YOUR_USERNAME/medpacs-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/medpacs-ai/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![DICOM](https://img.shields.io/badge/protocol-DICOM%20C--STORE-blue)](https://www.dicomstandard.org/)
[![NIfTI](https://img.shields.io/badge/output-NIfTI%20.nii.gz-green)](https://nifti.nimh.nih.gov/)

> **A production-grade DICOM-to-NIfTI ETL gateway with MRIQC-inspired automated QC and peer-reviewed normalization strategies — designed for clinical AI pipelines.**

Built as a portfolio project demonstrating deep expertise in medical imaging, DICOM standards, and the preprocessing techniques used in production at companies like **Siemens Healthineers**, **Philips**, and **Merantix Momentum**.

---

## What This Is

Most medical AI demos process pre-cleaned benchmark datasets. Real clinical pipelines are different: scanners send raw DICOM over the network, data has missing slices, inconsistent protocols, scanner-dependent intensities, and corrupted files. This project solves that.

```
[MRI Scanner / PACS]
       │
       │ DICOM C-STORE (port 11112)       ← Real clinical protocol
       ▼
┌─────────────────┐
│  C-STORE SCP    │  pynetdicom listener
└────────┬────────┘
         │  auto-triggers ETL when series complete
         ▼
┌──────────────────────────────────────────────────────────┐
│                    ETL PIPELINE                          │
│                                                          │
│  EXTRACT → QC GATE → TRANSFORM → LOAD                   │
│                                                          │
│  • DICOM magic-byte detection                            │
│  • Series grouping by SeriesInstanceUID                  │
│  • MRIQC-inspired IQMs: SNR, FBER, EFC, CJV             │
│  • nnU-Net foreground z-score normalization              │
│  • WhiteStripe tissue-anchored normalization             │
│  • Nyúl-Udupa population-scale histogram landmarks       │
│  • Voxel resampling to isotropic spacing                 │
│  • NIfTI affine from DICOM geometry tags                 │
│  • SQLite metadata tracking with upsert                  │
└──────────────────┬───────────────────────────────────────┘
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
      .nii.gz    .png     SQLite
      volumes  previews    DB
          │
          ├── FastAPI REST API  →  http://localhost:8000/docs
          ├── Flask Dashboard   →  http://localhost:5000
          └── C# .NET Client   →  dotnet run (MedPacsClient/)
```

---

## Key Features

| Feature | Details |
|---|---|
| **DICOM Networking** | C-STORE SCP on port 11112 — receives from any modality or PACS |
| **Automated QC** | MRIQC-inspired IQMs: SNR, FBER, EFC, CJV — bad scans never reach storage |
| **3 Normalization Strategies** | nnU-Net foreground z-score, WhiteStripe (WM-anchored), Nyúl-Udupa landmarks |
| **NIfTI Affine** | Proper 4×4 affine from `ImagePositionPatient` + `ImageOrientationPatient` |
| **REST API** | FastAPI with auto Swagger docs, Pydantic validation, background tasks |
| **Web Dashboard** | Dark-mode Flask UI with live log, preview images, pipeline run button |
| **C# .NET Client** | Console app consuming the REST API — demonstrates .NET 8 integration |
| **Docker** | Multi-stage build, docker-compose with all 4 services |
| **47 Unit Tests** | Pytest with coverage reporting, GitHub Actions CI on Python 3.11/3.12 |

---

## Research Implemented

This is not a toy pipeline. The normalization and QC strategies are directly pulled from peer-reviewed papers:

| Paper | Year | What is implemented |
|---|---|---|
| Isensee F. et al. — *Nature Methods* | 2021, 2024 | nnU-Net foreground-only z-score normalization |
| Shinohara R.T. et al. — *NeuroImage: Clinical* | 2014 | WhiteStripe WM-peak histogram normalization |
| Nyúl L.G. & Udupa J.K. — *IEEE TMI* | 1999, 2000 | Piecewise-linear histogram landmark standardization |
| Esteban O. et al. — *PLOS ONE* | 2017 | MRIQC image quality metrics (SNR, FBER, EFC, CJV) |

---

## Project Structure

```
medpacs_ai/
├── etl/
│   ├── models.py          SQLAlchemy ORM + typed dataclasses
│   ├── extractor.py       DICOM parser: magic bytes, series grouping, slice sorting
│   ├── qc.py              MRIQC-inspired automated quality control
│   ├── normalizer.py      3 research-backed normalization strategies
│   ├── transformer.py     Resampling, affine construction, preview extraction
│   ├── loader.py          NIfTI save, PNG preview, SQLite upsert
│   └── pipeline.py        Full orchestrator: Extract→QC→Transform→Load
├── api/
│   └── main.py            FastAPI REST API (Swagger at /docs)
├── dashboard/
│   └── app.py             Flask web dashboard
├── dicom_listener/
│   └── listener.py        DICOM C-STORE SCP (pynetdicom)
├── MedPacsClient/         C# .NET 8 console client
│   ├── MedPacsClient.csproj
│   ├── Program.cs
│   ├── ApiClient.cs
│   └── Models.cs
├── scripts/
│   ├── generate_synthetic_dicoms.py   Create test data
│   └── visualize_qc.py                QC + normalization visualization report
├── tests/
│   ├── test_extractor.py  (WIP)
│   ├── test_transformer.py
│   ├── test_normalizer.py
│   ├── test_qc.py
│   └── test_loader.py
├── .github/workflows/ci.yml   GitHub Actions CI
├── Dockerfile
├── docker-compose.yml
├── config.yaml
├── pyproject.toml
└── main.py                    CLI entry point
```

---

## Quick Start

### Option A — Local Python

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/medpacs-ai.git
cd medpacs-ai

# 2. Install
pip install -r requirements.txt

# 3. Generate synthetic DICOM test data
python scripts/generate_synthetic_dicoms.py

# 4. Run the ETL pipeline
python main.py

# 5. Start the web dashboard
python dashboard/app.py
# → http://localhost:5000
```

### Option B — Docker Compose (all services)

```bash
# Generate test data first
python scripts/generate_synthetic_dicoms.py

# Start all 4 services
docker-compose up --build

# Services:
#   http://localhost:5000  →  Web Dashboard
#   http://localhost:8000  →  REST API (Swagger at /docs)
#   port 11112             →  DICOM C-STORE SCP
```

### Option C — REST API only

```bash
pip install fastapi uvicorn
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs  (Swagger UI)
```

### Option D — C# .NET Client

```bash
cd MedPacsClient
dotnet run
# Requires the FastAPI server to be running on port 8000
```

---

## Sending DICOM via C-STORE (simulate a real modality)

Once the listener is running:

```bash
# Start the listener
python dicom_listener/listener.py --port 11112 --ae-title MEDPACS_SCP

# Use storescu (from DCMTK) to push a DICOM file
storescu -v 127.0.0.1 11112 path/to/your.dcm

# Or use pydicom's storescu
python -c "
from pynetdicom import AE
from pynetdicom.sop_class import CTImageStorage
ae = AE()
ae.add_requested_context(CTImageStorage)
assoc = ae.associate('127.0.0.1', 11112, ae_title='MEDPACS_SCP')
if assoc.is_established:
    import pydicom
    status = assoc.send_c_store(pydicom.dcmread('your_file.dcm'))
    print('Status:', hex(status.Status))
    assoc.release()
"
```

---

## Configuration

All pipeline parameters are in `config.yaml`:

```yaml
transform:
  normalization: "foreground_zscore"   # nnU-Net (recommended for MRI)
  # Options: z_score | foreground_zscore | whitestripe | min_max | window_level

qc:
  enabled: true
  min_snr:  5.0    # Signal-to-Noise Ratio
  min_fber: 3.0    # FMRIB Brain Extraction Ratio
  max_cjv:  2.0    # Coefficient of Joint Variation (lower = better)
```

---

## Running Tests

```bash
# All tests with coverage
python -m pytest tests/ -v --cov=etl --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
start htmlcov\index.html # Windows

# Lint
pip install ruff
ruff check etl/ tests/
```

---

## Technical Background

### Why foreground z-score over regular z-score?

MRI volumes are typically 70-80% background (air/padding). Standard z-score computes mean and std over ALL voxels, including this large background, which severely biases the normalization. nnU-Net's foreground-only approach (Isensee et al. 2021) computes statistics only over non-zero voxels, giving physiologically meaningful normalization.

### Why NIfTI affine matters

Most DICOM-to-NIfTI converters (dcm2niix) are offline tools. This pipeline constructs the 4×4 affine matrix from DICOM tags `ImagePositionPatient` and `ImageOrientationPatient` at runtime, preserving the patient coordinate system so that AI model predictions can be correctly mapped back to patient space.

### DICOM C-STORE vs folder watching

Many "PACS integration" demos just watch a folder. Real clinical workflows use DICOM networking (C-STORE protocol). This project implements an actual SCP that any DICOM-compliant system can push to — the same protocol used in hospitals every day.

---

## Author

**Muhammad Umer Raja**
M.Sc. Medical Image & Data Processing — Friedrich-Alexander-Universität Erlangen-Nürnberg
Software Engineer: C#, .NET, C++, Python, SQL

---

## License

MIT — see [LICENSE](LICENSE)
