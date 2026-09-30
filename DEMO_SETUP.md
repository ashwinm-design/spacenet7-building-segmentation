# SIH26012 — Live Demo Setup & Execution Guide

This document provides complete, reproducible instructions for launching the SIH26012 AI Building Footprint & Cadastral Feature Extraction system on Windows.

---

## 1. Prerequisites

- **Operating System:** Windows 10/11 (64-bit) or Linux/macOS
- **Python:** Python 3.10+ (tested and validated on Python 3.14.6)
- **Node.js:** Node.js v18+ (tested on Node v24.13.0 with npm 11.6.2)
- **Terminal:** PowerShell or Command Prompt (Windows)

---

## 2. Python Environment & Backend Setup

### Step 2.1: Clone & Navigate to Project Directory
```powershell
cd "c:\Users\Ashwinpc\OneDrive\Desktop\sih project"
```

### Step 2.2: (Optional) Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 2.3: Install Backend Demo Dependencies
```powershell
pip install -r requirements-demo.txt
```

---

## 3. Frontend WebGIS Setup

### Step 3.1: Navigate to Frontend Directory & Install Dependencies
```powershell
cd frontend
npm install
```

### Step 3.2: Verify Frontend Production Build
```powershell
npm run build
```
*(Build should complete with 0 errors in under 10 seconds)*

```powershell
cd ..
```

---

## 4. Model & Data Verification

Verify that the required model checkpoints and demo datasets are present in their target paths:

```powershell
python src/health_check.py
```

Expected output:
```
========================================
SIH26012 SYSTEM HEALTH CHECK
========================================
Python                  PASS
PyTorch                 PASS
Model checkpoint        PASS
Road DL checkpoint      PASS
Demo imagery            PASS
GeoPandas               PASS
Rasterio                PASS
Shapely                 PASS
Backend                 PASS
Frontend build          PASS

Overall:
READY
```

### Required File Locations
1. **Building Footprint Model:** `results/checkpoints/best_model.pth` (164.15 MB)
2. **Road Deep Learning Model:** `results/experiments/road_dl/checkpoints/best_road_model.pth` (54.76 MB)
3. **Demo Imagery:** `data/open_cities_raw/ptn/f49f31/f49f31.tif` (110.67 MB)
4. **Generalization Imagery:** `data/open_cities_raw/ptn/abe1a3/abe1a3.tif` (232.88 MB)

---

## 5. Starting the Live System

### Terminal 1: Launch Backend REST API
Run the one-command PowerShell startup script:
```powershell
.\scripts\start_backend.ps1
```
Or directly via uvicorn:
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
- API Health Check URL: `http://127.0.0.1:8000/api/health`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

### Terminal 2: Launch Frontend WebGIS
Run the one-command PowerShell startup script:
```powershell
.\scripts\start_frontend.ps1
```
Or directly via npm:
```powershell
cd frontend
npm run dev
```
- WebGIS Application URL: `http://localhost:5173`

---

## 6. Live Presentation Workflow (5-Minute Sequence)

1. **Open Browser:** Navigate to `http://localhost:5173`.
2. **Launch AI Analysis:**
   - Click **"Run Open Cities Demo"** (ROI: 3072, 2048, 4096, 3072).
   - Observe the real-time 10-stage checklist progressing dynamically from `QUEUED` $\rightarrow$ `COMPLETED` (~38s CPU).
3. **Open WebGIS:** Click `[ OPEN WEBGIS ]` once processing completes.
4. **Inspect Layers:**
   - Toggle **Building Footprints** (Orange/Coral polygons, 254 detected).
   - Toggle **Hybrid Road Corridors** (Cyan/Teal access corridors, 1,571m).
   - Toggle **Preliminary Parcels** (Amber road-constrained polygons).
5. **Verify Parcel Candidates:**
   - In the **Parcel Verification** queue, click candidate **#56** (lowest confidence: 62.54%).
   - Inspect multi-source conflict flag and zero road frontage. Click **"Needs Review"**.
   - Select candidate **#159** (100.0% confidence, clear frontage). Click **"Accept"**.
   - Notice the KPI badges update live: *Accepted: 1, Needs Review: 1*.
6. **Export GIS Deliverables:**
   - In the **GIS Export** panel, download:
     - `Reviewed Parcels (GeoJSON)`
     - `Hybrid Roads (GeoJSON)`
     - `Analysis Summary (JSON)`
     - `Review Audit Log (JSON)`

---

## 7. Automated Headless Demo Command

To execute and verify the complete workflow headless from the terminal:
```powershell
python src/run_demo.py
```

---

## 8. Resetting Demo State

To clean up demo-generated analyses while keeping all models, datasets, and benchmarks intact:
```powershell
python src/reset_demo.py --dry-run
python src/reset_demo.py --confirm
```
*(Reference run `8b8c724c7e` is permanently protected and never deleted)*
