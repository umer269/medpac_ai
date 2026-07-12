# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2024-07-12

### Added
- **ETL Pipeline** — Full Extract → QC → Transform → Load orchestrator
- **DICOM Extractor** — Magic-byte detection, series grouping by `SeriesInstanceUID`, slice sorting by `ImagePositionPatient`, rescale slope/intercept application
- **MRIQC-inspired QC Gate** — Automated quality control computing SNR, FBER, EFC, CJV; rejects scout scans, corrupted volumes, and low-quality acquisitions before processing
- **Research-backed Normalizer module** (`etl/normalizer.py`):
  - `foreground_zscore` — nnU-Net v2 strategy (Isensee et al., *Nature Methods* 2021/2024)
  - `whitestripe_normalize` — WM-peak tissue-anchored normalization (Shinohara et al., *NeuroImage: Clinical* 2014)
  - `NyulUdupaNormalizer` — Piecewise-linear histogram landmark standardization with `fit/transform/save/load` API (Nyúl & Udupa, *IEEE TMI* 1999/2000)
- **DICOM Transformer** — scipy voxel resampling to isotropic spacing, NIfTI affine from `ImagePositionPatient`/`ImageOrientationPatient`, 5 normalization options
- **Loader** — Compressed NIfTI `.nii.gz` output, PNG preview slice, SQLite upsert with full metadata
- **DICOM C-STORE SCP** — pynetdicom listener on port 11112 accepting DICOM from any modality or PACS, with 30s debounced auto-trigger of ETL
- **FastAPI REST API** (`api/main.py`) — Full CRUD over processed series, async pipeline trigger, auto Swagger/OpenAPI docs at `/docs`
- **Flask Web Dashboard** (`dashboard/app.py`) — Dark-mode dashboard with live log, preview images, stats, and Run Pipeline button
- **C# .NET 8 Console Client** (`MedPacsClient/`) — Interactive CLI consuming the REST API with Spectre.Console tables, retry logic, CSV export
- **Docker** — Multi-stage `Dockerfile` + `docker-compose.yml` orchestrating all 4 services
- **GitHub Actions CI** — Automated test runs on Python 3.11/3.12, ruff lint, Docker build validation
- **47 unit tests** — Full pytest coverage for transformer, normalizer (20 tests), QC (10 tests), and loader
- **Synthetic DICOM generator** (`scripts/generate_synthetic_dicoms.py`) — Creates realistic brain MRI + CT abdomen test data without any external downloads
- **QC Visualization Report** (`scripts/visualize_qc.py`) — Publication-quality figures: normalization comparison, intensity histograms, QC gauge charts, MIP projections
- **pyproject.toml** — ruff, mypy strict, pytest, coverage configuration

### Technical Highlights
- First-class NIfTI affine construction from DICOM geometry tags (supports AI prediction mapping back to patient space)
- All data contracts use typed Python dataclasses — no dict-passing between stages
- SQLAlchemy ORM enables future migration to PostgreSQL for enterprise deployments
- WhiteStripe and Nyúl-Udupa normalizers include `save/load` persistence so population-scale statistics survive across pipeline runs
