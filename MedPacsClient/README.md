# MedPACS-AI Console Client

> A professional, interactive .NET 8 console client for the **MedPACS-AI** DICOM Intelligence Platform REST API.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Rich UI** | All output rendered with [Spectre.Console](https://spectreconsole.net/) – tables, panels, progress bars, spinners |
| **Health Check** | Startup banner displays live API status, version, uptime and GPU info |
| **Series Browser** | Paginated table with Patient, Modality, Study Date, Slices, Spacing, Pipeline Status, Size |
| **Series Detail** | Full metadata breakdown including extended DICOM tags |
| **Pipeline Monitor** | Submit jobs and watch live progress bars poll every 2 s until completion |
| **Statistics Dashboard** | Catalogue totals, modality breakdown, pipeline status breakdown |
| **CSV Export** | One-click export of the full series catalogue to `medpacs_export.csv` |
| **Retry Logic** | Transparent exponential back-off with jitter (3 retries) |
| **Cancellation** | `Ctrl+C` cleanly cancels any in-flight request |

---

## 📋 Prerequisites

| Requirement | Minimum Version |
|---|---|
| [.NET SDK](https://dotnet.microsoft.com/download/dotnet/8.0) | **8.0** |
| MedPACS-AI API server | Running at `http://localhost:8000` |
| Terminal | Any ANSI-colour capable terminal (Windows Terminal, iTerm2, etc.) |

---

## 🚀 Getting Started

### 1 – Clone / navigate to the project

```bash
cd MedPacsClient
```

### 2 – Restore packages

```bash
dotnet restore
```

### 3 – Run (development)

```bash
dotnet run
```

The client will automatically connect to `http://localhost:8000`.  
To point to a different server, set the environment variable:

```bash
# Windows PowerShell
$env:MEDPACS_URL = "http://192.168.1.100:8000"
dotnet run

# Linux / macOS
MEDPACS_URL=http://192.168.1.100:8000 dotnet run
```

### 4 – Build a self-contained executable (optional)

```bash
# Windows x64
dotnet publish -c Release -r win-x64 --self-contained true -o ./publish

# Linux x64
dotnet publish -c Release -r linux-x64 --self-contained true -o ./publish
```

---

## 🗂 Project Structure

```
MedPacsClient/
├── MedPacsClient.csproj   # Project file (net8.0, Spectre.Console, DI)
├── Program.cs             # Entry point + MedPacsApp controller (menu, rendering)
├── ApiClient.cs           # MedPacsApiClient (HTTP, retry, cancellation)
├── Models.cs              # C# record types matching FastAPI Pydantic models
└── README.md              # This file
```

---

## 🎨 UI Preview

### Startup Banner

```
  __  __          _ ____   _    ____ ____       _    ___ 
 |  \/  | ___  __| |  _ \ / \  / ___/ ___|     / \  |_ _|
 | |\/| |/ _ \/ _` | |_) / _ \| |  \___ \    / _ \  | | 
 | |  | |  __/ (_| |  __/ ___ \ |___ ___) |  / ___ \ | | 
 |_|  |_|\___|\__,_|_| /_/   \_\____|____/  /_/   \_\___|

──────────── DICOM Intelligence Platform  •  Console Client v1.0 ──────────────

╭─ Service Health ──────────────────────────────────╮
│  Status  :  HEALTHY                               │
│  Version :  1.0.0                                 │
│  Uptime  :  2h 14m 07s                            │
│  DICOM   :  348 series loaded                     │
│  GPU     :  ✔ Available                           │
╰───────────────────────────────────────────────────╯
```

### Series List Table

```
╭─ DICOM Series Catalogue (348 series) ─────────────────────────────────────────────────────╮
│  #  │ Patient              │ Modality │ Study Date │ Description          │ Slices │  Size │
├─────┼──────────────────────┼──────────┼────────────┼──────────────────────┼────────┼───────┤
│   1 │ SMITH^JOHN (P00123)  │    CT    │ 2024-11-12 │ Chest w/ Contrast    │    320 │ 1.2GB │
│   2 │ JONES^MARY (P00456)  │    MR    │ 2024-11-13 │ Brain T1 MPRAGE      │    176 │ 498MB │
│   3 │ DOE^JANE   (P00789)  │    PT    │ 2024-11-14 │ Whole Body FDG       │    219 │ 862MB │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
```

### Pipeline Progress Bar

```
Pipeline processing  ████████████████████████░░░░░░  78%  00:01:42  ⠇
Stage: Running segmentation model (layer 4/5)
```

---

## 🔌 API Endpoints Used

| Menu Option | HTTP Method | Endpoint |
|---|---|---|
| Health check (startup) | `GET` | `/health` |
| List all series | `GET` | `/series` |
| View series detail | `GET` | `/series/{uid}` |
| Run pipeline | `POST` | `/pipeline/run` |
| Poll pipeline status | `GET` | `/pipeline/status/{job_id}` |
| Show statistics | `GET` | `/stats` |

---

## 🛡 Error Handling

- **Transient network failures** are retried up to 3 times with exponential back-off (500 ms → 1 s → 2 s + ±10 % jitter).
- **Non-transient API errors** (4xx / 5xx) are displayed immediately in a red error panel without retrying.
- **Ctrl+C** at any prompt or during polling cleanly aborts the current operation and returns to the menu (or exits if at the menu itself).

---

## 📄 License

This project is part of the **MedPACS-AI** portfolio. All rights reserved.
