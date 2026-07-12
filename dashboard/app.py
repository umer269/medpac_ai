"""
dashboard/app.py
─────────────────
Flask web dashboard for the MedPACS-AI ETL Pipeline.

Provides:
  GET  /                   — main dashboard
  GET  /api/series         — JSON list of all processed series from DB
  GET  /api/preview/<uid>  — serve preview PNG for a series
  GET  /api/logs           — last 50 lines of pipeline.log
  POST /api/run            — trigger the ETL pipeline (async)
  GET  /api/run/status     — get pipeline run status
"""

import os
import base64
import subprocess
import threading
import time
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from flask import Flask, jsonify, send_file, abort, render_template_string

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
DB_PATH     = BASE_DIR / "output" / "metadata.db"
LOG_PATH    = BASE_DIR / "output" / "pipeline.log"
PREVIEW_DIR = BASE_DIR / "output" / "previews"
NIFTI_DIR   = BASE_DIR / "output" / "nifti"

app = Flask(__name__)

# Pipeline run state
_pipeline_state = {
    "running": False,
    "last_exit_code": None,
    "last_run_at": None,
    "output": [],
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_db_session():
    if not DB_PATH.exists():
        return None
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
    return Session(engine)


def run_pipeline_async():
    _pipeline_state["running"] = True
    _pipeline_state["output"]  = []
    _pipeline_state["last_exit_code"] = None
    _pipeline_state["last_run_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    proc = subprocess.Popen(
        ["python", "main.py"],
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:
        _pipeline_state["output"].append(line.rstrip())
    proc.wait()
    _pipeline_state["last_exit_code"] = proc.returncode
    _pipeline_state["running"] = False


# ─── API routes ───────────────────────────────────────────────────────────────

@app.route("/api/series")
def api_series():
    session = get_db_session()
    if session is None:
        return jsonify([])

    rows = session.execute(text("""
        SELECT series_uid, patient_id, patient_name, modality,
               study_date, study_desc, num_slices, rows, cols,
               pixel_spacing_x, pixel_spacing_y, slice_thickness,
               nifti_path, preview_path, status, error_message,
               processed_at
        FROM dicom_studies
        ORDER BY processed_at DESC
    """)).fetchall()
    session.close()

    result = []
    for r in rows:
        nifti_size_mb = None
        if r.nifti_path and os.path.exists(r.nifti_path):
            nifti_size_mb = round(os.path.getsize(r.nifti_path) / 1_048_576, 2)

        result.append({
            "series_uid":       r.series_uid,
            "patient_id":       r.patient_id,
            "patient_name":     str(r.patient_name).replace("^", " "),
            "modality":         r.modality,
            "study_date":       r.study_date,
            "study_desc":       r.study_desc,
            "num_slices":       r.num_slices,
            "rows":             r.rows,
            "cols":             r.cols,
            "pixel_spacing_x":  r.pixel_spacing_x,
            "pixel_spacing_y":  r.pixel_spacing_y,
            "slice_thickness":  r.slice_thickness,
            "nifti_size_mb":    nifti_size_mb,
            "status":           r.status,
            "error_message":    r.error_message,
            "processed_at":     str(r.processed_at),
        })
    return jsonify(result)


@app.route("/api/preview/<path:series_uid>")
def api_preview(series_uid):
    safe_uid = series_uid.replace(".", "_")
    # Search for preview PNG matching this series
    matches = list(PREVIEW_DIR.glob(f"{safe_uid}_preview.png"))
    if not matches:
        abort(404)
    return send_file(str(matches[0]), mimetype="image/png")


@app.route("/api/logs")
def api_logs():
    if not LOG_PATH.exists():
        return jsonify({"lines": []})
    with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return jsonify({"lines": [l.rstrip() for l in lines[-60:]]})


@app.route("/api/run", methods=["POST"])
def api_run():
    if _pipeline_state["running"]:
        return jsonify({"status": "already_running"}), 409
    t = threading.Thread(target=run_pipeline_async, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/run/status")
def api_run_status():
    return jsonify({
        "running":        _pipeline_state["running"],
        "last_exit_code": _pipeline_state["last_exit_code"],
        "last_run_at":    _pipeline_state["last_run_at"],
        "output":         _pipeline_state["output"][-30:],
    })


@app.route("/api/stats")
def api_stats():
    session = get_db_session()
    if session is None:
        return jsonify({"total": 0, "success": 0, "failed": 0, "modalities": {}})

    rows = session.execute(text("""
        SELECT status, modality, COUNT(*) as cnt
        FROM dicom_studies
        GROUP BY status, modality
    """)).fetchall()
    session.close()

    total = success = failed = 0
    modalities = {}
    for r in rows:
        total += r.cnt
        if r.status == "success":
            success += r.cnt
        else:
            failed += r.cnt
        modalities[r.modality] = modalities.get(r.modality, 0) + r.cnt

    return jsonify({
        "total": total,
        "success": success,
        "failed": failed,
        "modalities": modalities,
    })


# ─── Main dashboard page ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


# ─── Dashboard HTML ───────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>MedPACS-AI | ETL Pipeline Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --bg:         #0a0e1a;
      --surface:    #111827;
      --surface2:   #1a2235;
      --border:     #1e2d45;
      --accent:     #3b82f6;
      --accent-glow:#1d4ed8;
      --green:      #10b981;
      --red:        #ef4444;
      --yellow:     #f59e0b;
      --purple:     #8b5cf6;
      --cyan:       #06b6d4;
      --text:       #e2e8f0;
      --muted:      #64748b;
      --font:       'Inter', sans-serif;
      --mono:       'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--font);
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }

    /* ── Header ───────────────────────────────────────────── */
    header {
      background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
      border-bottom: 1px solid var(--border);
      padding: 20px 32px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky; top: 0; z-index: 100;
      backdrop-filter: blur(10px);
    }
    .logo { display: flex; align-items: center; gap: 14px; }
    .logo-icon {
      width: 44px; height: 44px; border-radius: 12px;
      background: linear-gradient(135deg, var(--accent), var(--purple));
      display: flex; align-items: center; justify-content: center;
      font-size: 22px; box-shadow: 0 0 24px rgba(59,130,246,0.4);
    }
    .logo-text h1 { font-size: 1.2rem; font-weight: 700; letter-spacing: -0.3px; }
    .logo-text p  { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }
    .header-right { display: flex; align-items: center; gap: 12px; }
    .badge {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 5px 12px; border-radius: 20px; font-size: 0.72rem; font-weight: 600;
      border: 1px solid; text-transform: uppercase; letter-spacing: 0.5px;
    }
    .badge-blue   { background: rgba(59,130,246,0.1);  border-color: rgba(59,130,246,0.3); color: var(--accent); }
    .badge-green  { background: rgba(16,185,129,0.1);  border-color: rgba(16,185,129,0.3); color: var(--green); }
    .badge-yellow { background: rgba(245,158,11,0.1);  border-color: rgba(245,158,11,0.3); color: var(--yellow); }

    /* ── Main Layout ──────────────────────────────────────── */
    main { max-width: 1400px; margin: 0 auto; padding: 32px; }

    /* ── Stats Row ────────────────────────────────────────── */
    .stats-grid {
      display: grid; grid-template-columns: repeat(4, 1fr);
      gap: 16px; margin-bottom: 32px;
    }
    .stat-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 22px 24px;
      transition: transform 0.2s, box-shadow 0.2s;
      position: relative; overflow: hidden;
    }
    .stat-card:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(0,0,0,0.4); }
    .stat-card::before {
      content: ''; position: absolute;
      top: 0; left: 0; right: 0; height: 3px;
      border-radius: 16px 16px 0 0;
    }
    .stat-card.blue::before   { background: linear-gradient(90deg, var(--accent), var(--cyan)); }
    .stat-card.green::before  { background: linear-gradient(90deg, var(--green), #34d399); }
    .stat-card.purple::before { background: linear-gradient(90deg, var(--purple), #a78bfa); }
    .stat-card.yellow::before { background: linear-gradient(90deg, var(--yellow), #fbbf24); }
    .stat-icon { font-size: 1.6rem; margin-bottom: 10px; }
    .stat-value { font-size: 2rem; font-weight: 700; letter-spacing: -1px; }
    .stat-label { font-size: 0.78rem; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }

    /* ── Pipeline Diagram ─────────────────────────────────── */
    .pipeline-section {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 28px;
    }
    .section-title {
      font-size: 0.85rem; font-weight: 600; color: var(--muted);
      text-transform: uppercase; letter-spacing: 1px;
      margin-bottom: 20px; display: flex; align-items: center; gap: 8px;
    }
    .pipeline-flow {
      display: flex; align-items: center; justify-content: space-between;
      gap: 8px; flex-wrap: wrap;
    }
    .pipe-step {
      flex: 1; min-width: 140px;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 12px; padding: 16px 18px;
      text-align: center; position: relative;
    }
    .pipe-step-icon { font-size: 1.8rem; margin-bottom: 8px; }
    .pipe-step-name { font-size: 0.8rem; font-weight: 700; color: var(--text); }
    .pipe-step-desc { font-size: 0.68rem; color: var(--muted); margin-top: 4px; line-height: 1.4; }
    .pipe-step-tag {
      display: inline-block; font-size: 0.6rem; font-weight: 600;
      padding: 2px 7px; border-radius: 20px; margin-top: 8px;
      background: rgba(59,130,246,0.15); color: var(--accent);
      border: 1px solid rgba(59,130,246,0.3);
    }
    .pipe-arrow { font-size: 1.4rem; color: var(--muted); flex-shrink: 0; }

    /* ── Two-column layout ────────────────────────────────── */
    .two-col { display: grid; grid-template-columns: 1fr 380px; gap: 24px; align-items: start; }

    /* ── Series Cards ─────────────────────────────────────── */
    .series-container { display: flex; flex-direction: column; gap: 16px; }
    .series-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px; overflow: hidden;
      transition: box-shadow 0.2s, border-color 0.2s;
    }
    .series-card:hover { border-color: rgba(59,130,246,0.4); box-shadow: 0 4px 24px rgba(59,130,246,0.1); }
    .series-inner { display: grid; grid-template-columns: 180px 1fr; gap: 0; }
    .series-preview {
      background: #000;
      display: flex; align-items: center; justify-content: center;
      min-height: 180px; position: relative;
    }
    .series-preview img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .series-preview .no-preview {
      color: var(--muted); font-size: 0.75rem; text-align: center; padding: 16px;
    }
    .modality-badge {
      position: absolute; top: 8px; left: 8px;
      background: rgba(0,0,0,0.8); border: 1px solid var(--border);
      border-radius: 6px; padding: 3px 8px;
      font-size: 0.7rem; font-weight: 700; color: var(--cyan);
      font-family: var(--mono);
    }
    .series-info { padding: 20px 22px; }
    .series-name { font-size: 1rem; font-weight: 600; margin-bottom: 4px; }
    .series-uid  { font-size: 0.65rem; color: var(--muted); font-family: var(--mono); margin-bottom: 14px; }
    .meta-grid   { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px; }
    .meta-item label { display: block; font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }
    .meta-item span  { font-size: 0.82rem; font-weight: 500; }
    .iqm-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
    .iqm-chip {
      background: var(--surface2); border: 1px solid var(--border);
      border-radius: 8px; padding: 5px 10px;
      font-size: 0.7rem; font-family: var(--mono);
    }
    .iqm-chip span { color: var(--cyan); font-weight: 600; }
    .status-row { display: flex; align-items: center; gap: 8px; margin-top: 12px; }
    .dot { width: 8px; height: 8px; border-radius: 50%; }
    .dot-green  { background: var(--green);  box-shadow: 0 0 6px var(--green); }
    .dot-red    { background: var(--red);    box-shadow: 0 0 6px var(--red); }
    .dot-yellow { background: var(--yellow); box-shadow: 0 0 6px var(--yellow); }

    /* ── Right Panel ──────────────────────────────────────── */
    .right-panel { display: flex; flex-direction: column; gap: 20px; }
    .panel-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px; padding: 20px;
    }
    .run-btn {
      width: 100%; padding: 14px; border-radius: 12px;
      background: linear-gradient(135deg, var(--accent), var(--purple));
      border: none; color: #fff; font-family: var(--font);
      font-size: 0.9rem; font-weight: 600; cursor: pointer;
      transition: opacity 0.2s, transform 0.2s;
      display: flex; align-items: center; justify-content: center; gap: 8px;
      box-shadow: 0 4px 20px rgba(59,130,246,0.3);
    }
    .run-btn:hover { opacity: 0.9; transform: translateY(-1px); }
    .run-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .run-output {
      margin-top: 14px; background: #000;
      border: 1px solid var(--border); border-radius: 10px;
      padding: 12px; max-height: 200px; overflow-y: auto;
      font-family: var(--mono); font-size: 0.65rem; color: #94a3b8;
      line-height: 1.6; display: none;
    }
    .run-output.visible { display: block; }

    /* ── Log Panel ────────────────────────────────────────── */
    .log-output {
      background: #000; border: 1px solid var(--border);
      border-radius: 10px; padding: 12px;
      max-height: 260px; overflow-y: auto;
      font-family: var(--mono); font-size: 0.65rem; color: #94a3b8; line-height: 1.7;
    }
    .log-line.success { color: #34d399; }
    .log-line.warning { color: #fbbf24; }
    .log-line.error   { color: #f87171; }
    .log-line.info    { color: #93c5fd; }

    /* ── Spinner ──────────────────────────────────────────── */
    .spinner {
      width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.3);
      border-top-color: #fff; border-radius: 50%;
      animation: spin 0.8s linear infinite; display: none;
    }
    .spinner.active { display: inline-block; }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Empty state ──────────────────────────────────────── */
    .empty-state {
      text-align: center; padding: 60px 20px;
      color: var(--muted); border: 2px dashed var(--border);
      border-radius: 16px;
    }
    .empty-state .icon { font-size: 3rem; margin-bottom: 12px; }

    /* ── Scrollbar ────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
  </style>
</head>
<body>
  <header>
    <div class="logo">
      <div class="logo-icon">🫁</div>
      <div class="logo-text">
        <h1>MedPACS-AI</h1>
        <p>Medical Imaging ETL Pipeline Dashboard</p>
      </div>
    </div>
    <div class="header-right">
      <span class="badge badge-blue">DICOM → NIfTI</span>
      <span class="badge badge-green" id="db-status">● DB Connected</span>
      <span class="badge badge-yellow">v1.0.0</span>
    </div>
  </header>

  <main>
    <!-- Stats -->
    <div class="stats-grid">
      <div class="stat-card blue">
        <div class="stat-icon">📂</div>
        <div class="stat-value" id="stat-total">–</div>
        <div class="stat-label">Series Processed</div>
      </div>
      <div class="stat-card green">
        <div class="stat-icon">✅</div>
        <div class="stat-value" id="stat-success">–</div>
        <div class="stat-label">Successfully Loaded</div>
      </div>
      <div class="stat-card purple">
        <div class="stat-icon">🔬</div>
        <div class="stat-value" id="stat-modalities">–</div>
        <div class="stat-label">Modalities</div>
      </div>
      <div class="stat-card yellow">
        <div class="stat-icon">🛡️</div>
        <div class="stat-value" id="stat-qc">Active</div>
        <div class="stat-label">QC Gate (MRIQC)</div>
      </div>
    </div>

    <!-- Pipeline Architecture -->
    <div class="pipeline-section">
      <div class="section-title">⚡ Pipeline Architecture</div>
      <div class="pipeline-flow">
        <div class="pipe-step">
          <div class="pipe-step-icon">🗂️</div>
          <div class="pipe-step-name">EXTRACT</div>
          <div class="pipe-step-desc">DICOM magic-byte detection, series grouping, slice sorting, rescale slope</div>
          <span class="pipe-step-tag">pydicom</span>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
          <div class="pipe-step-icon">🛡️</div>
          <div class="pipe-step-name">QC GATE</div>
          <div class="pipe-step-desc">SNR, FBER, EFC, CJV — MRIQC-inspired automated quality control</div>
          <span class="pipe-step-tag">Esteban et al. 2017</span>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
          <div class="pipe-step-icon">🔬</div>
          <div class="pipe-step-name">TRANSFORM</div>
          <div class="pipe-step-desc">Foreground z-score (nnU-Net), WhiteStripe, Nyúl-Udupa, voxel resampling</div>
          <span class="pipe-step-tag">Isensee 2021 · Shinohara 2014</span>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
          <div class="pipe-step-icon">💾</div>
          <div class="pipe-step-name">LOAD</div>
          <div class="pipe-step-desc">NIfTI .nii.gz, PNG preview, SQLite metadata DB with upsert</div>
          <span class="pipe-step-tag">nibabel · SQLAlchemy</span>
        </div>
      </div>
    </div>

    <!-- Main content -->
    <div class="two-col">
      <!-- Left: processed series -->
      <div>
        <div class="section-title" style="margin-bottom:16px;">📋 Processed Series</div>
        <div class="series-container" id="series-container">
          <div class="empty-state">
            <div class="icon">🔍</div>
            <p>Loading series from database…</p>
          </div>
        </div>
      </div>

      <!-- Right panel -->
      <div class="right-panel">
        <!-- Run Pipeline -->
        <div class="panel-card">
          <div class="section-title">🚀 Run Pipeline</div>
          <p style="font-size:0.78rem; color:var(--muted); margin-bottom:16px; line-height:1.5;">
            Trigger the ETL pipeline to process all DICOM files in the inbox folder.
          </p>
          <button class="run-btn" id="run-btn" onclick="runPipeline()">
            <div class="spinner" id="run-spinner"></div>
            <span id="run-btn-text">▶  Run Pipeline</span>
          </button>
          <div class="run-output" id="run-output"></div>
        </div>

        <!-- Live log -->
        <div class="panel-card">
          <div class="section-title">📄 Pipeline Log</div>
          <div class="log-output" id="log-output">
            <span style="color:var(--muted)">Loading logs…</span>
          </div>
        </div>

        <!-- Research papers -->
        <div class="panel-card">
          <div class="section-title">📚 Research Used</div>
          <div style="display:flex;flex-direction:column;gap:10px;">
            <div style="background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:12px;">
              <div style="font-size:0.72rem;font-weight:600;color:var(--cyan)">nnU-Net Foreground Z-Score</div>
              <div style="font-size:0.67rem;color:var(--muted);margin-top:3px;">Isensee et al. — Nature Methods 2021/2024</div>
            </div>
            <div style="background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:12px;">
              <div style="font-size:0.72rem;font-weight:600;color:var(--cyan)">WhiteStripe Normalization</div>
              <div style="font-size:0.67rem;color:var(--muted);margin-top:3px;">Shinohara et al. — NeuroImage Clinical 2014</div>
            </div>
            <div style="background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:12px;">
              <div style="font-size:0.72rem;font-weight:600;color:var(--cyan)">Nyúl–Udupa Landmarks</div>
              <div style="font-size:0.67rem;color:var(--muted);margin-top:3px;">Nyúl & Udupa — IEEE TMI 1999/2000</div>
            </div>
            <div style="background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:12px;">
              <div style="font-size:0.72rem;font-weight:600;color:var(--cyan)">MRIQC Image Quality Metrics</div>
              <div style="font-size:0.67rem;color:var(--muted);margin-top:3px;">Esteban et al. — PLOS ONE 2017</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </main>

  <script>
    async function fetchStats() {
      try {
        const r = await fetch('/api/stats');
        const d = await r.json();
        document.getElementById('stat-total').textContent = d.total;
        document.getElementById('stat-success').textContent = d.success;
        const mods = Object.keys(d.modalities).join(' · ') || '–';
        document.getElementById('stat-modalities').textContent = mods;
      } catch(e) {}
    }

    async function fetchSeries() {
      try {
        const r = await fetch('/api/series');
        const series = await r.json();
        const container = document.getElementById('series-container');

        if (series.length === 0) {
          container.innerHTML = `
            <div class="empty-state">
              <div class="icon">📭</div>
              <p>No series processed yet.<br>Click "Run Pipeline" to get started.</p>
            </div>`;
          return;
        }

        container.innerHTML = series.map(s => {
          const uid_safe = s.series_uid;
          const modColor = s.modality === 'MR' ? '#8b5cf6' : s.modality === 'CT' ? '#06b6d4' : '#f59e0b';
          const statusDot = s.status === 'success'
            ? '<span class="dot dot-green"></span>'
            : '<span class="dot dot-red"></span>';
          const spacing = s.pixel_spacing_x ? `${s.pixel_spacing_x.toFixed(2)} × ${s.pixel_spacing_y.toFixed(2)} mm` : '–';
          const size = s.nifti_size_mb ? `${s.nifti_size_mb} MB` : '–';

          return `
          <div class="series-card">
            <div class="series-inner">
              <div class="series-preview">
                <img src="/api/preview/${uid_safe}"
                     alt="Preview"
                     onerror="this.parentElement.innerHTML='<div class=no-preview>No preview</div>'"/>
                <div class="modality-badge">${s.modality}</div>
              </div>
              <div class="series-info">
                <div class="series-name">${s.patient_name || 'Unknown Patient'}</div>
                <div class="series-uid">${s.series_uid.substring(0, 40)}…</div>
                <div class="meta-grid">
                  <div class="meta-item"><label>Study</label><span>${s.study_desc || '–'}</span></div>
                  <div class="meta-item"><label>Patient ID</label><span>${s.patient_id || '–'}</span></div>
                  <div class="meta-item"><label>Slices</label><span>${s.num_slices} × ${s.rows}×${s.cols}px</span></div>
                  <div class="meta-item"><label>Spacing</label><span>${spacing}</span></div>
                  <div class="meta-item"><label>Thickness</label><span>${s.slice_thickness ? s.slice_thickness + ' mm' : '–'}</span></div>
                  <div class="meta-item"><label>NIfTI Size</label><span>${size}</span></div>
                </div>
                <div class="status-row">
                  ${statusDot}
                  <span style="font-size:0.75rem; color: ${s.status==='success' ? 'var(--green)' : 'var(--red)'}">
                    ${s.status === 'success' ? 'Loaded Successfully' : 'Failed: ' + (s.error_message||'Unknown')}
                  </span>
                  <span style="font-size:0.68rem;color:var(--muted);margin-left:auto">${s.processed_at ? s.processed_at.substring(0,19) : ''}</span>
                </div>
              </div>
            </div>
          </div>`;
        }).join('');
      } catch(e) {
        console.error(e);
      }
    }

    async function fetchLogs() {
      try {
        const r = await fetch('/api/logs');
        const d = await r.json();
        const el = document.getElementById('log-output');
        el.innerHTML = d.lines.map(line => {
          let cls = '';
          if (line.includes('SUCCESS') || line.includes('complete')) cls = 'success';
          else if (line.includes('WARNING') || line.includes('WARN'))  cls = 'warning';
          else if (line.includes('ERROR'))  cls = 'error';
          else if (line.includes('INFO'))   cls = 'info';
          return `<div class="log-line ${cls}">${line}</div>`;
        }).join('') || '<span style="color:var(--muted)">No logs yet.</span>';
        el.scrollTop = el.scrollHeight;
      } catch(e) {}
    }

    let runPolling = null;
    async function runPipeline() {
      const btn     = document.getElementById('run-btn');
      const spinner = document.getElementById('run-spinner');
      const btnText = document.getElementById('run-btn-text');
      const output  = document.getElementById('run-output');

      btn.disabled = true;
      spinner.classList.add('active');
      btnText.textContent = 'Running…';
      output.classList.add('visible');
      output.textContent = 'Starting pipeline…\n';

      await fetch('/api/run', { method: 'POST' });

      runPolling = setInterval(async () => {
        const r = await fetch('/api/run/status');
        const d = await r.json();
        output.textContent = d.output.join('\n') || 'Running…';
        output.scrollTop = output.scrollHeight;

        if (!d.running) {
          clearInterval(runPolling);
          btn.disabled = false;
          spinner.classList.remove('active');
          btnText.textContent = d.last_exit_code === 0 ? '✓  Done!' : '✗  Error';
          setTimeout(() => { btnText.textContent = '▶  Run Pipeline'; }, 3000);
          fetchStats();
          fetchSeries();
          fetchLogs();
        }
      }, 800);
    }

    // Initial load
    fetchStats();
    fetchSeries();
    fetchLogs();

    // Auto-refresh logs every 10s
    setInterval(fetchLogs, 10000);
  </script>
</body>
</html>"""


if __name__ == "__main__":
    print("MedPACS-AI Dashboard running at: http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)
