"""
api/main.py
──────────────────────────────────────────────────────────────────────────────
MedPACS-AI REST API — built with FastAPI.

Exposes the ETL pipeline data as a production-grade REST API with:
  - Auto-generated OpenAPI / Swagger documentation  →  /docs
  - Pydantic validation models for all responses
  - CORS enabled for web dashboard
  - Background task support for async pipeline runs

Endpoints:
  GET  /health                  — service health check
  GET  /series                  — list all processed series
  GET  /series/{uid}            — single series full metadata
  GET  /series/{uid}/preview    — PNG preview image
  POST /pipeline/run            — trigger ETL pipeline (async)
  GET  /pipeline/status         — pipeline run status
  GET  /stats                   — aggregate statistics

Run:
  uvicorn api.main:app --reload --port 8000
  → OpenAPI docs at http://localhost:8000/docs
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent.parent
DB_PATH     = Path(os.environ.get("MEDPACS_OUTPUT_DIR", BASE_DIR / "output")) / "metadata.db"
PREVIEW_DIR = Path(os.environ.get("MEDPACS_OUTPUT_DIR", BASE_DIR / "output")) / "previews"
LOG_PATH    = Path(os.environ.get("MEDPACS_OUTPUT_DIR", BASE_DIR / "output")) / "pipeline.log"

# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "MedPACS-AI REST API",
    description = (
        "Clinical-grade DICOM ETL pipeline REST interface.\n\n"
        "Implements DICOM → NIfTI conversion with MRIQC-inspired QC gating "
        "and research-backed normalization strategies (nnU-Net, WhiteStripe, "
        "Nyúl–Udupa). Built for integration with enterprise PACS and C# .NET viewers."
    ),
    version     = "1.0.0",
    contact     = {
        "name":  "Muhammad Umer Raja",
        "email": "umer.raja@medpacs.dev",
    },
    license_info = {
        "name": "MIT",
        "url":  "https://opensource.org/licenses/MIT",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pipeline run state ───────────────────────────────────────────────────────

_state: Dict[str, Any] = {
    "running":        False,
    "last_exit_code": None,
    "last_run_at":    None,
    "output":         [],
}


# ─── Pydantic models ─────────────────────────────────────────────────────────

class SeriesSummary(BaseModel):
    series_uid:      str
    patient_id:      Optional[str]
    patient_name:    Optional[str]
    modality:        Optional[str]
    study_date:      Optional[str]
    study_desc:      Optional[str]
    num_slices:      Optional[int]
    rows:            Optional[int]
    cols:            Optional[int]
    pixel_spacing_x: Optional[float]
    pixel_spacing_y: Optional[float]
    slice_thickness: Optional[float]
    nifti_path:      Optional[str]
    preview_path:    Optional[str]
    status:          Optional[str]
    error_message:   Optional[str]
    processed_at:    Optional[str]
    nifti_size_mb:   Optional[float]


class PipelineStatus(BaseModel):
    running:        bool
    last_exit_code: Optional[int]
    last_run_at:    Optional[str]
    output:         List[str]


class Stats(BaseModel):
    total:      int
    success:    int
    failed:     int
    modalities: Dict[str, int]


class HealthResponse(BaseModel):
    status:    str
    db_exists: bool
    version:   str


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_session() -> Optional[Session]:
    if not DB_PATH.exists():
        return None
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    return Session(engine)


def _run_pipeline_bg():
    _state["running"]        = True
    _state["output"]         = []
    _state["last_exit_code"] = None
    _state["last_run_at"]    = time.strftime("%Y-%m-%dT%H:%M:%S")
    proc = subprocess.Popen(
        [sys.executable, str(BASE_DIR / "main.py")],
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:
        _state["output"].append(line.rstrip())
    proc.wait()
    _state["last_exit_code"] = proc.returncode
    _state["running"]        = False


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Service health check — use this for Docker HEALTHCHECK and load-balancer probes."""
    return HealthResponse(
        status    = "ok",
        db_exists = DB_PATH.exists(),
        version   = "1.0.0",
    )


@app.get("/series", response_model=List[SeriesSummary], tags=["Series"])
def list_series():
    """Return all processed DICOM series from the metadata database, newest first."""
    session = get_session()
    if session is None:
        return []
    rows = session.execute(text("""
        SELECT * FROM dicom_studies ORDER BY processed_at DESC
    """)).fetchall()
    session.close()

    result = []
    for r in rows:
        nifti_mb = None
        if r.nifti_path and os.path.exists(r.nifti_path):
            nifti_mb = round(os.path.getsize(r.nifti_path) / 1_048_576, 2)
        result.append(SeriesSummary(
            series_uid      = r.series_uid,
            patient_id      = r.patient_id,
            patient_name    = str(r.patient_name).replace("^", " ") if r.patient_name else None,
            modality        = r.modality,
            study_date      = r.study_date,
            study_desc      = r.study_desc,
            num_slices      = r.num_slices,
            rows            = r.rows,
            cols            = r.cols,
            pixel_spacing_x = r.pixel_spacing_x,
            pixel_spacing_y = r.pixel_spacing_y,
            slice_thickness = r.slice_thickness,
            nifti_path      = r.nifti_path,
            preview_path    = r.preview_path,
            status          = r.status,
            error_message   = r.error_message,
            processed_at    = str(r.processed_at),
            nifti_size_mb   = nifti_mb,
        ))
    return result


@app.get("/series/{series_uid}", response_model=SeriesSummary, tags=["Series"])
def get_series(series_uid: str):
    """Return full metadata for a single DICOM series by its SeriesInstanceUID."""
    session = get_session()
    if session is None:
        raise HTTPException(status_code=503, detail="Database not available")
    row = session.execute(text(
        "SELECT * FROM dicom_studies WHERE series_uid = :uid"
    ), {"uid": series_uid}).fetchone()
    session.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Series '{series_uid}' not found")

    nifti_mb = None
    if row.nifti_path and os.path.exists(row.nifti_path):
        nifti_mb = round(os.path.getsize(row.nifti_path) / 1_048_576, 2)

    return SeriesSummary(
        series_uid      = row.series_uid,
        patient_id      = row.patient_id,
        patient_name    = str(row.patient_name).replace("^", " ") if row.patient_name else None,
        modality        = row.modality,
        study_date      = row.study_date,
        study_desc      = row.study_desc,
        num_slices      = row.num_slices,
        rows            = row.rows,
        cols            = row.cols,
        pixel_spacing_x = row.pixel_spacing_x,
        pixel_spacing_y = row.pixel_spacing_y,
        slice_thickness = row.slice_thickness,
        nifti_path      = row.nifti_path,
        preview_path    = row.preview_path,
        status          = row.status,
        error_message   = row.error_message,
        processed_at    = str(row.processed_at),
        nifti_size_mb   = nifti_mb,
    )


@app.get("/series/{series_uid}/preview", tags=["Series"])
def get_preview(series_uid: str):
    """
    Return the PNG axial preview slice for a series.
    Suitable for direct embedding in `<img>` tags.
    """
    safe_uid = series_uid.replace(".", "_")
    matches  = list(PREVIEW_DIR.glob(f"{safe_uid}_preview.png"))
    if not matches:
        raise HTTPException(status_code=404, detail="Preview not found for this series")
    return FileResponse(str(matches[0]), media_type="image/png")


@app.post("/pipeline/run", tags=["Pipeline"])
def run_pipeline(background_tasks: BackgroundTasks):
    """
    Trigger the ETL pipeline asynchronously.
    Poll `/pipeline/status` for progress and completion.
    """
    if _state["running"]:
        raise HTTPException(status_code=409, detail="Pipeline is already running")
    background_tasks.add_task(_run_pipeline_bg)
    return {"status": "started", "message": "Poll /pipeline/status for progress"}


@app.get("/pipeline/status", response_model=PipelineStatus, tags=["Pipeline"])
def pipeline_status():
    """Return the current pipeline run status and recent stdout output."""
    return PipelineStatus(
        running        = _state["running"],
        last_exit_code = _state["last_exit_code"],
        last_run_at    = _state["last_run_at"],
        output         = _state["output"][-50:],
    )


@app.get("/stats", response_model=Stats, tags=["System"])
def stats():
    """Return aggregate statistics over all processed series."""
    session = get_session()
    if session is None:
        return Stats(total=0, success=0, failed=0, modalities={})
    rows = session.execute(text("""
        SELECT status, modality, COUNT(*) as cnt
        FROM dicom_studies GROUP BY status, modality
    """)).fetchall()
    session.close()

    total = success = failed = 0
    modalities: Dict[str, int] = {}
    for r in rows:
        total   += r.cnt
        success += r.cnt if r.status == "success" else 0
        failed  += r.cnt if r.status != "success" else 0
        modalities[r.modality] = modalities.get(r.modality, 0) + r.cnt

    return Stats(total=total, success=success, failed=failed, modalities=modalities)
