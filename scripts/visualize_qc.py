"""
scripts/visualize_qc.py
──────────────────────────────────────────────────────────────────────────────
QC Metrics Visualization — generates a detailed HTML + PNG report showing:

  1. Normalization comparison:  raw vs z_score vs foreground_zscore vs whitestripe
  2. Intensity histogram overlay for each normalization strategy
  3. QC metrics (SNR, FBER, EFC, CJV) as gauge charts
  4. 3D MIP (Maximum Intensity Projection) — axial, sagittal, coronal

Run from project root:
    python scripts/visualize_qc.py

Outputs:
    output/qc_report/normalization_comparison.png
    output/qc_report/intensity_histograms.png
    output/qc_report/qc_metrics.png
    output/qc_report/mip_projection.png
    output/qc_report/report.html
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent.parent))
from etl.normalizer import foreground_zscore, whitestripe_normalize, NyulUdupaNormalizer
from etl.qc import SeriesQCChecker
from etl.models import DicomSeries

OUT_DIR = Path("output/qc_report")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Synthetic MRI generator ──────────────────────────────────────────────────

def make_brain_volume(shape=(40, 128, 128), seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vol = np.zeros(shape, dtype=np.float32)
    cx, cy = shape[1] // 2, shape[2] // 2
    Y, X = np.ogrid[:shape[1], :shape[2]]
    for z in range(shape[0]):
        r = int(shape[1] * 0.40 * (1 - 0.25 * abs(z / shape[0] - 0.5)))
        dist = np.sqrt(((X - cx) / (r + 1)) ** 2 + ((Y - cy) / (r + 1)) ** 2)
        sl = np.zeros((shape[1], shape[2]), dtype=np.float32)
        sl[dist < 1.0] = 400 + rng.normal(0, 30, (shape[1], shape[2]))[dist < 1.0]
        sl[dist < 0.7] = 700 + rng.normal(0, 35, (shape[1], shape[2]))[dist < 0.7]
        sl[dist < 0.3] = 100 + rng.normal(0, 20, (shape[1], shape[2]))[dist < 0.3]
        sl = np.clip(sl, 0, None)
        vol[z] = sl
    return vol

# ─── 1. Normalization Comparison ─────────────────────────────────────────────

def plot_normalization_comparison(vol: np.ndarray):
    print("  Generating normalization comparison...")

    # Nyul-Udupa: train on 5 augmented versions
    norm = NyulUdupaNormalizer()
    rng = np.random.default_rng(0)
    for i in range(5):
        norm.update(vol * rng.uniform(0.7, 1.3))
    norm.fit()

    strategies = {
        "Raw (no normalization)":     vol,
        "Z-Score (whole volume)":     (vol - vol.mean()) / (vol.std() + 1e-8),
        "Foreground Z-Score\n(nnU-Net, Isensee 2021)": foreground_zscore(vol),
        "WhiteStripe\n(Shinohara 2014)":               whitestripe_normalize(vol),
        "Nyúl-Udupa Landmarks\n(IEEE TMI 2000)":       norm.transform(vol),
    }

    mid_z = vol.shape[0] // 2
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.patch.set_facecolor("#0a0e1a")

    for ax, (name, v) in zip(axes, strategies.items()):
        slice_ = v[mid_z]
        im = ax.imshow(slice_, cmap="gray", interpolation="bilinear")
        ax.set_title(name, color="white", fontsize=8, pad=6, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(ax, "xticklabels"), color="white")

    fig.suptitle("Normalization Strategy Comparison — Mid-brain Axial Slice",
                 color="white", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = OUT_DIR / "normalization_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
    plt.close()
    print(f"    Saved: {path}")

# ─── 2. Intensity Histograms ──────────────────────────────────────────────────

def plot_intensity_histograms(vol: np.ndarray):
    print("  Generating intensity histogram overlay...")

    norm = NyulUdupaNormalizer()
    rng = np.random.default_rng(1)
    for i in range(5):
        norm.update(vol * rng.uniform(0.7, 1.3))
    norm.fit()

    strategies = {
        "Raw":              vol,
        "Whole-vol Z-Score":(vol - vol.mean()) / (vol.std() + 1e-8),
        "Foreground Z-Score": foreground_zscore(vol),
        "WhiteStripe":      whitestripe_normalize(vol),
        "Nyúl-Udupa":       norm.transform(vol),
    }
    colors = ["#94a3b8", "#3b82f6", "#10b981", "#8b5cf6", "#f59e0b"]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#111827")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.spines[:].set_color("#1e2d45")

    for (name, v), color in zip(strategies.items(), colors):
        fg = v[v != 0].flatten()
        fg = fg[(fg > np.percentile(fg, 1)) & (fg < np.percentile(fg, 99))]
        ax.hist(fg, bins=200, alpha=0.65, color=color, label=name, density=True)

    ax.legend(facecolor="#1a2235", edgecolor="#1e2d45", labelcolor="white", fontsize=9)
    ax.set_xlabel("Intensity", color="white")
    ax.set_ylabel("Density", color="white")
    ax.set_title("Foreground Intensity Distribution by Normalization Strategy",
                 color="white", fontweight="bold", fontsize=12)

    plt.tight_layout()
    path = OUT_DIR / "intensity_histograms.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
    plt.close()
    print(f"    Saved: {path}")

# ─── 3. QC Metrics Gauges ────────────────────────────────────────────────────

def plot_qc_metrics(vol: np.ndarray):
    print("  Generating QC metrics visualization...")

    series = DicomSeries(
        series_uid="1.2.3", study_uid="1", patient_id="P001",
        patient_name="Demo", modality="MR", study_date="20240601",
        study_desc="Brain MRI", num_slices=vol.shape[0],
        rows=vol.shape[1], cols=vol.shape[2],
        pixel_spacing=(1.0, 1.0), slice_thickness=3.0, pixel_array=vol,
    )
    checker = SeriesQCChecker(min_snr=1.0, min_fber=0.1, max_cjv=5.0)
    report = checker.check(series)

    metrics = {
        "SNR": (report.snr,  0, 50,  15, "Signal-to-Noise Ratio"),
        "FBER": (report.fber, 0, 100, 10, "FMRIB Brain Extraction Ratio"),
        "EFC": (abs(report.efc), 0, 2, 1,  "Entropy Focus Criterion"),
        "CJV": (report.cjv,  0, 3,  0.5, "Coefficient of Joint Variation"),
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor("#0a0e1a")

    for ax, (metric, (val, vmin, vmax, threshold, desc)) in zip(axes, metrics.items()):
        ax.set_facecolor("#111827")
        normalized = min(max((val - vmin) / (vmax - vmin), 0), 1)
        color = "#10b981" if (metric != "CJV" and val >= threshold) or \
                             (metric == "CJV" and val <= threshold) else "#ef4444"

        theta = np.linspace(np.pi, 0, 200)
        ax.plot(np.cos(theta), np.sin(theta), color="#1e2d45", lw=10, solid_capstyle="round")
        theta_fill = np.linspace(np.pi, np.pi - normalized * np.pi, 200)
        ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=color, lw=10, solid_capstyle="round")

        ax.text(0, -0.1, f"{val:.2f}", ha="center", va="center",
                fontsize=18, fontweight="bold", color=color)
        ax.text(0, -0.45, metric, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        ax.text(0, -0.65, desc, ha="center", va="center",
                fontsize=7, color="#64748b", wrap=True)
        ax.text(0, -0.82, f"Threshold: {threshold}", ha="center",
                fontsize=7, color="#64748b")

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.0, 1.1)
        ax.axis("off")

    fig.suptitle("MRIQC-Inspired Image Quality Metrics (IQMs)",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "qc_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
    plt.close()
    print(f"    Saved: {path}")

# ─── 4. MIP Projection ───────────────────────────────────────────────────────

def plot_mip_projection(vol: np.ndarray):
    print("  Generating MIP projections...")
    v = foreground_zscore(vol)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0a0e1a")

    mip_cmap = LinearSegmentedColormap.from_list("mip", ["#0a0e1a", "#1d4ed8", "#06b6d4", "#ffffff"])

    views = [
        ("Axial MIP",    np.max(v, axis=0)),
        ("Sagittal MIP", np.max(v, axis=1)),
        ("Coronal MIP",  np.max(v, axis=2)),
    ]
    for ax, (title, mip) in zip(axes, views):
        ax.imshow(mip, cmap=mip_cmap, interpolation="bilinear")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Maximum Intensity Projection (MIP) — Foreground Z-Score Normalized",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "mip_projection.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
    plt.close()
    print(f"    Saved: {path}")

# ─── 5. HTML Report ───────────────────────────────────────────────────────────

def generate_html_report():
    print("  Generating HTML report...")
    images = {
        "Normalization Comparison":   "normalization_comparison.png",
        "Intensity Histograms":        "intensity_histograms.png",
        "QC Metrics (MRIQC IQMs)":     "qc_metrics.png",
        "MIP Projections":             "mip_projection.png",
    }
    cards = ""
    for title, fname in images.items():
        cards += f"""
        <div class="card">
          <h2>{title}</h2>
          <img src="{fname}" alt="{title}"/>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>MedPACS-AI QC Report</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet"/>
  <style>
    body {{ font-family: 'Inter', sans-serif; background: #0a0e1a; color: #e2e8f0; margin: 0; padding: 32px; }}
    h1 {{ font-size: 1.8rem; margin-bottom: 4px; }}
    .sub {{ color: #64748b; margin-bottom: 40px; }}
    .card {{ background: #111827; border: 1px solid #1e2d45; border-radius: 16px;
             padding: 24px; margin-bottom: 24px; }}
    .card h2 {{ font-size: 1rem; margin-bottom: 16px; color: #06b6d4; }}
    .card img {{ width: 100%; border-radius: 8px; }}
    .badge {{ display: inline-block; background: rgba(59,130,246,0.15);
              border: 1px solid rgba(59,130,246,0.3); color: #3b82f6;
              padding: 4px 12px; border-radius: 20px; font-size: 0.75rem;
              font-weight: 600; margin-right: 8px; margin-bottom: 24px; }}
  </style>
</head>
<body>
  <h1>MedPACS-AI — QC & Normalization Report</h1>
  <p class="sub">Generated by the automated QC pipeline</p>
  <span class="badge">nnU-Net Normalization</span>
  <span class="badge">WhiteStripe (Shinohara 2014)</span>
  <span class="badge">Nyúl-Udupa (IEEE TMI 2000)</span>
  <span class="badge">MRIQC IQMs (Esteban 2017)</span>
  {cards}
</body>
</html>"""

    path = OUT_DIR / "report.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"    Saved: {path}")

# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nMedPACS-AI QC Visualization Report\n" + "=" * 40)
    vol = make_brain_volume()
    print(f"Generated synthetic brain volume: {vol.shape}  min={vol.min():.0f}  max={vol.max():.0f}")
    plot_normalization_comparison(vol)
    plot_intensity_histograms(vol)
    plot_qc_metrics(vol)
    plot_mip_projection(vol)
    generate_html_report()
    print(f"\nDone! Open: {OUT_DIR / 'report.html'}\n")
