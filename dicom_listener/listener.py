"""
dicom_listener/listener.py
──────────────────────────────────────────────────────────────────────────────
DICOM C-STORE SCP (Service Class Provider) — the clinical gateway node.

This turns MedPACS-AI into a real PACS-compatible endpoint.
Any DICOM modality (MRI scanner, CT, PACS workstation) can push images
directly to this listener using standard DICOM networking, and the ETL
pipeline will automatically process them.

Supported DICOM Services:
  - C-ECHO  (DICOM ping / connectivity verification)
  - C-STORE (receive DICOM instances from any SCU)

How it works:
  1. A modality or PACS (the SCU) establishes a DICOM association.
  2. It sends C-STORE requests (one per DICOM file).
  3. This SCP saves each file to the inbox and responds with Success (0x0000).
  4. When a complete series is detected (no new files for 30s), the ETL
     pipeline is triggered automatically.

Usage:
  python dicom_listener/listener.py
  python dicom_listener/listener.py --port 11112 --ae-title MEDPACS_SCP

References:
  - DICOM Standard PS3.7 — Message Exchange (C-STORE)
  - DICOM Standard PS3.4 — Service Class Specifications
  - pynetdicom documentation: https://pydicom.github.io/pynetdicom/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from loguru import logger

try:
    from pynetdicom import AE, evt, AllStoragePresentationContexts, VerificationPresentationContexts
    from pynetdicom.sop_class import Verification
except ImportError:
    print("ERROR: pynetdicom not installed. Run: pip install pynetdicom")
    sys.exit(1)

import pydicom

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).parent.parent
INBOX_DIR = Path(os.environ.get("MEDPACS_DATA_DIR", BASE_DIR / "data")) / "dicom_inbox"
INBOX_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_AE_TITLE = os.environ.get("DICOM_AE_TITLE", "MEDPACS_SCP")
DEFAULT_PORT     = int(os.environ.get("DICOM_PORT", "11112"))

# ─── Pipeline trigger (debounced) ─────────────────────────────────────────────

class _PipelineTrigger:
    """
    Debounced pipeline trigger.
    Waits IDLE_SECONDS after the last received file before launching the ETL,
    so a series is fully received before processing starts.
    """
    IDLE_SECONDS = 30

    def __init__(self):
        self._timer: threading.Timer | None = None
        self._lock  = threading.Lock()

    def reset(self):
        with self._lock:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.IDLE_SECONDS, self._fire)
            self._timer.daemon = True
            self._timer.start()
        logger.debug(f"Pipeline trigger reset (fires in {self.IDLE_SECONDS}s of inactivity).")

    def _fire(self):
        logger.info("No new DICOM files received for 30s — launching ETL pipeline …")
        try:
            result = subprocess.run(
                [sys.executable, str(BASE_DIR / "main.py")],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.success("ETL pipeline completed successfully.")
            else:
                logger.error(f"ETL pipeline failed:\n{result.stderr}")
        except Exception as exc:
            logger.error(f"Failed to launch ETL pipeline: {exc}")


_trigger = _PipelineTrigger()

# ─── DICOM Event Handlers ─────────────────────────────────────────────────────

def handle_store(event) -> int:
    """
    Handle a C-STORE request.

    Called once per DICOM instance received.
    Saves the file to the inbox and resets the pipeline trigger.

    Returns:
        0x0000 (Success) — tells the SCU the instance was accepted.
    """
    ds = event.dataset
    ds.file_meta = event.file_meta

    # Build a safe filename: SeriesUID / SOPInstanceUID.dcm
    series_uid   = getattr(ds, "SeriesInstanceUID", "UNKNOWN_SERIES")
    sop_uid      = getattr(ds, "SOPInstanceUID",    "UNKNOWN_SOP")
    modality     = getattr(ds, "Modality",          "XX")
    patient_id   = getattr(ds, "PatientID",         "ANON")

    series_dir   = INBOX_DIR / str(series_uid).replace(".", "_")
    series_dir.mkdir(parents=True, exist_ok=True)

    filename = series_dir / f"{str(sop_uid).replace('.', '_')}.dcm"
    pydicom.dcmwrite(str(filename), ds)

    logger.info(
        f"[C-STORE] Received {modality} instance — "
        f"Patient={patient_id}  Series={str(series_uid)[:20]}…  "
        f"→ {filename.name}"
    )

    _trigger.reset()
    return 0x0000   # DICOM Status: Success


def handle_echo(event) -> int:
    """
    Handle a C-ECHO request (DICOM ping).
    Returns Success — confirms the SCP is reachable.
    """
    calling_ae = event.assoc.requestor.ae_title
    logger.info(f"[C-ECHO] Ping from AE: '{calling_ae}' — responding Success.")
    return 0x0000


# ─── Build & start the Application Entity ────────────────────────────────────

def start_listener(ae_title: str, port: int) -> None:
    ae = AE(ae_title=ae_title)

    # Accept ALL storage presentation contexts (MR, CT, CR, US, NM, PT …)
    ae.supported_contexts = (
        AllStoragePresentationContexts
        + VerificationPresentationContexts
    )

    handlers = [
        (evt.EVT_C_STORE, handle_store),
        (evt.EVT_C_ECHO,  handle_echo),
    ]

    logger.info("=" * 60)
    logger.info(f"MedPACS-AI  DICOM C-STORE SCP")
    logger.info(f"  AE Title : {ae_title}")
    logger.info(f"  Port     : {port}")
    logger.info(f"  Inbox    : {INBOX_DIR}")
    logger.info(f"  Auto ETL : {_trigger.IDLE_SECONDS}s after last file")
    logger.info("=" * 60)
    logger.info("Waiting for DICOM associations …  (Ctrl+C to stop)")

    # Blocking server — handles associations on a thread pool
    ae.start_server(
        ("0.0.0.0", port),
        block=True,
        evt_handlers=handlers,
    )


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MedPACS-AI DICOM C-STORE SCP listener"
    )
    parser.add_argument("--port",     type=int, default=DEFAULT_PORT,     help="DICOM port (default 11112)")
    parser.add_argument("--ae-title", type=str, default=DEFAULT_AE_TITLE, help="AE title (default MEDPACS_SCP)")
    args = parser.parse_args()

    start_listener(ae_title=args.ae_title, port=args.port)
