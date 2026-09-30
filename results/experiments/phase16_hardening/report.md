# Phase 16 — SIH Demo Hardening & Presentation Readiness Report

**Project:** SIH26012 — High-Resolution Aerial & Drone Building Footprint Segmentation and Cadastral Feature Extraction  
**Date:** September 30, 2026  
**Status:** Hardened, Fully Tested & Demo-Ready  

---

## 1. Executive Summary & Objective

Phase 16 represents the final hardening and presentation readiness milestone for the SIH26012 prototype. In adherence to strict engineering constraints:
- **No machine learning models were retrained.**
- **No model architectures were altered.**
- **No frozen checkpoints were modified.**
- **No benchmark metrics were fabricated.**
- **Reference prototype analysis `8b8c724c7e` was preserved 100% untouched.**

The primary objective was to transform the validated multi-phase pipeline (Phases 13A–13D, 14A, 14B, 15) into an easily demonstrable, thoroughly documented, and fault-tolerant system ready for live demonstration before the Smart India Hackathon (SIH) jury.

---

## 2. Environment & Runtime Audit

The system execution environment was audited and benchmarked on Windows 11 (AMD64):

| Component | Audited Version / Setting | Status |
| :--- | :--- | :---: |
| **Operating System** | Windows 11 (Kernel 10.0.26200-SP0) | PASS |
| **Python Runtime** | Python 3.14.6 AMD64 | PASS |
| **Node.js & npm** | Node v24.13.0 / npm 11.6.2 | PASS |
| **Deep Learning** | PyTorch 2.14.0 (CPU mode benchmarked) | PASS |
| **Geospatial Stack** | GeoPandas 1.2.0, Rasterio 1.5.1, Shapely 2.1.2, PyProj 3.8.0, PyOgrio 0.13.0 | PASS |
| **Computer Vision** | OpenCV 5.0.0.93, Pillow 12.3.0, Scikit-Image 0.26.0 | PASS |
| **Backend Service** | FastAPI 0.141.1, Uvicorn 0.54.0 on `http://127.0.0.1:8000` | PASS |
| **WebGIS Frontend** | Vite 5.4.21, React 18.3.1, MapLibre GL 4.7.1 on `http://localhost:5173` | PASS |
| **Frontend Build** | `tsc && vite build` compiled 1,755 modules in 7.89s with 0 errors | PASS |

Detailed configuration: [`results/experiments/phase16_hardening/environment_audit.md`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/results/experiments/phase16_hardening/environment_audit.md)  
Streamlined requirements: [`requirements-demo.txt`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/requirements-demo.txt)

---

## 3. Model Checkpoints Audit & Verification

All frozen model artifacts were verified for physical existence, size, and cryptographic SHA-256 integrity:

| Model Description | Checkpoint File | Size | SHA-256 Hash | Verification Status |
| :--- | :--- | :---: | :--- | :---: |
| **3-Class Building Footprint Model** | `results/checkpoints/best_model.pth` | 164.15 MB | `529654f5d5e934c9eef71539821a26736982116b2ab897729f95cadef51870c6` | **FROZEN & VERIFIED** |
| **Deep Learning Road Model** | `results/experiments/road_dl/checkpoints/best_road_model.pth` | 54.76 MB | `4e53c158d6ee77ea415d0c064d04d0b0404678fe110c696ca6cb7e61132ccd18` | **FROZEN & VERIFIED** |
| **SpaceNet 7 3-Class Baseline** | `results/experiments/3class_model/best_model.pth` | 164.15 MB | `3aa0736c7430c3b31755cc5afc0576b196cc7b6292eae0c57cd907fc9b97dbed` | **FROZEN & VERIFIED** |
| **Open Cities Adaptation Model** | `results/experiments/open_cities_adaptation/checkpoints/best_model.pth` | 54.77 MB | `ca8e80544ffd1b5bce4a9abe897a0136d46bd5e3da372974acbd3d38a1a37528` | **FROZEN & VERIFIED** |

Deliverable manifest: [`results/experiments/phase16_hardening/model_manifest.json`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/results/experiments/phase16_hardening/model_manifest.json)

---

## 4. Data Dependencies Audit

The data assets were audited and cataloged into three transparent tiers:
1. **Required for Live Demo:**
   - Primary Demo Tile: `data/open_cities_raw/ptn/f49f31/f49f31.tif` (11.07 MB, GeoTIFF, EPSG:32732, ROI 3072, 2048, 4096, 3072)
   - Generalization Tile: `data/open_cities_raw/ptn/abe1a3/abe1a3.tif` (232.88 MB, GeoTIFF, EPSG:32732, ROI 3000, 3000, 4024, 4024)
2. **Optional / Research:**
   - Ground-truth building polygons (`f49f31.geojson`, `abe1a3.geojson`)
   - Road patch benchmarks and SpaceNet 7 training sample archive
3. **Generated at Runtime:**
   - Exactly 17 isolated artifacts in `results/prototype/api/{analysis_id}/`

Deliverable manifest: [`results/experiments/phase16_hardening/demo_data_manifest.json`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/results/experiments/phase16_hardening/demo_data_manifest.json)

---

## 5. Startup Mechanisms & One-Command Automation

To eliminate manual multi-step shell operations during presentations, dedicated startup scripts were created for Windows (PowerShell & Batch):

- **Backend Startup:** [`scripts/start_backend.ps1`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/scripts/start_backend.ps1) & [`scripts/start_backend.bat`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/scripts/start_backend.bat)
  - Configures `PYTHONIOENCODING=utf-8` and unbuffered stdout.
  - Launches FastAPI ASGI server at `http://127.0.0.1:8000`.
- **Frontend Startup:** [`scripts/start_frontend.ps1`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/scripts/start_frontend.ps1) & [`scripts/start_frontend.bat`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/scripts/start_frontend.bat)
  - Launches Vite dev server at `http://localhost:5173`.
- **System Health Check Script:** [`src/health_check.py`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/src/health_check.py)
  - Verifies 10 core subsystems (Python, PyTorch, checkpoints, imagery, GIS libraries, backend, and frontend build).
  - Terminal output: 10/10 PASS $\rightarrow$ **Overall: READY**.
  - Output record: [`results/experiments/phase16_hardening/health_check.json`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/results/experiments/phase16_hardening/health_check.json)
- **Demo State Reset Script:** [`src/reset_demo.py`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/src/reset_demo.py)
  - Safely cleans demo analysis directories (`--dry-run` or `--confirm`) while strictly excluding `8b8c724c7e`.

---

## 6. Live Demo Repeatability & Stability Verification

Three consecutive end-to-end demo runs were executed on the primary demo subregion (`f49f31`, ROI 3072, 2048, 4096, 3072) using fresh, isolated analysis IDs:

| Metric | Run 1 (`repeat_run_1`) | Run 2 (`repeat_run_2`) | Run 3 (`repeat_run_3`) | Consistency |
| :--- | :---: | :---: | :---: | :---: |
| **Status** | `completed` | `completed` | `completed` | **100% Success** |
| **Total CPU Runtime** | **40.44 s** | **38.65 s** | **39.82 s** | Mean: 39.64 s ($\pm$0.9 s) |
| **Building Footprints** | 254 | 254 | 254 | **Identical (254)** |
| **Road Corridors** | 4 | 4 | 4 | **Identical (4)** |
| **Total Road Length** | 1,571.88 m | 1,571.88 m | 1,571.88 m | **Identical (1,571.88 m)** |
| **Preliminary Parcels** | 254 | 254 | 254 | **Identical (254)** |
| **Geometry Validity** | 100.0% | 100.0% | 100.0% | **100.0% Valid** |
| **Mean Confidence** | 88.17% | 88.17% | 88.17% | **Identical (88.17%)** |
| **Confidence Range** | [62.54% – 100.0%] | [62.54% – 100.0%] | [62.54% – 100.0%] | **Identical** |
| **Flagged for Review** | 210 | 210 | 210 | **Identical (210)** |

Deliverable record: [`results/experiments/phase16_hardening/repeatability_test.json`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/results/experiments/phase16_hardening/repeatability_test.json)

---

## 7. Deliverables & Export Verification

All 7 export deliverables generated during the demo were validated for structural and geometric integrity:

| Deliverable Artifact | Format | Feature Count | Geometry Integrity | Verification Result |
| :--- | :--- | :---: | :---: | :---: |
| **`buildings.geojson`** | GeoJSON FeatureCollection | 254 features | All geometries valid, EPSG:32732 | **PASS** |
| **`roads_hybrid.geojson`** | GeoJSON FeatureCollection | 4 corridor features | Valid multi-polygons, no self-intersections | **PASS** |
| **`parcels_hybrid.geojson`** | GeoJSON FeatureCollection | 254 candidates | 100.0% valid polygons, no nulls | **PASS** |
| **`reviewed_parcels.geojson`** | GeoJSON FeatureCollection | 254 candidates | Review status & confidence enriched | **PASS** |
| **`statistics.csv`** | CSV Table | 254 records | 10 attribute columns, no NaN/Inf | **PASS** |
| **`analysis_summary.json`** | JSON Object | Comprehensive KPIs | Contains stage timings and disclaimer | **PASS** |
| **`parcel_reviews_audit.json`**| JSON Array | Chronological log | Valid timestamped entries, hashes present | **PASS** |

Deliverable record: [`results/experiments/phase16_hardening/export_validation.json`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/results/experiments/phase16_hardening/export_validation.json)

---

## 8. Error Handling & Input Resilience

The REST API and WebGIS frontend handle edge cases gracefully with human-readable error messages and zero exposed tracebacks:

1. **Unsupported File Format:** Uploading `.txt` returns HTTP 400 with message: `"Unsupported file extension '.txt'. Allowed extensions: .jpeg, .jpg, .png, .tif, .tiff"`.
2. **Empty File Upload:** Uploading 0-byte file returns HTTP 400 with message: `"Uploaded file is empty (0 bytes)."`.
3. **Non-Existent Analysis ID:** Querying missing ID returns HTTP 404 with message: `"Analysis ID 'nonexistent999' not found."`.
4. **Invalid Review Status:** Submitting unrecognized status returns HTTP 400 with allowed enumeration values.
5. **Geometry Edit Validation:** When a user draws or edits a polygon in WebGIS, Shapely `make_valid` repairs self-intersections automatically prior to persistence.

---

## 9. Performance Baseline & Profiling

```
                     Stage Latency Distribution (f49f31 Demo Tile, CPU)
┌───────────────────────────────┬──────────────┬────────────┐
│ Pipeline Stage                │ Duration (s) │ % of Total │
├───────────────────────────────┼──────────────┼────────────┤
│ 1. Input Validation           │   0.10 s     │   0.26%    │
│ 2. Preprocessing & Tiling     │   0.41 s     │   1.06%    │
│ 3. Building AI Inference      │   1.46 s     │   3.76%    │
│ 4. Building Polygonization    │   3.44 s     │   8.86%    │
│ 5. Road Extraction (CV + DL)  │   3.33 s     │   8.58%    │
│ 6. Road Hybrid Fusion         │   6.21 s     │  16.00%    │
│ 7. Parcel Generation (Voronoi)│  23.17 s     │  59.70%    │ <── Primary Bottleneck
│ 8. Parcel Intelligence        │   0.51 s     │   1.31%    │
│ 9. Ready for Review           │   0.17 s     │   0.44%    │
├───────────────────────────────┼──────────────┼────────────┤
│ TOTAL PIPELINE RUNTIME        │  38.80 s     │ 100.00%    │
└───────────────────────────────┴──────────────┴────────────┘
```

Deliverable record: [`results/experiments/phase16_hardening/stage_timings.json`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/results/experiments/phase16_hardening/stage_timings.json)

---

## 10. Final Architecture Diagram

The high-resolution visual architecture diagram was generated and placed at:
[`results/experiments/phase16_hardening/final_architecture.png`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/results/experiments/phase16_hardening/final_architecture.png)

It illustrates:
1. **Data & Ingestion Layer:** GeoTIFF orthomosaic input, raster validation, and tiling.
2. **AI & Computer Vision Layer:** SpaceNet 7 3-class segmentation, polygonization, classical CV, and DL road models.
3. **Spatial Fusion & Parcel Engine Layer:** Phase 13D hybrid road fusion, road-constrained Voronoi partitioning, and Parcel Intelligence scoring.
4. **WebGIS & Human-in-the-Loop Layer:** FastAPI async backend, React/MapLibre WebGIS, candidate review, vertex editing, and audit logging.
5. **Standardized GIS Exports:** GeoJSON footprints, hybrid road corridors, reviewed parcels, and CSV/JSON audit trails.

---

## 11. Presentation Deliverables & Jury Guides

- **5-Minute Live Presentation Script:** [`results/experiments/phase16_hardening/SIH_DEMO_SCRIPT.md`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/results/experiments/phase16_hardening/SIH_DEMO_SCRIPT.md)  
  *(Structured second-by-second walkthrough from problem statement to GIS export)*
- **Judge Talking Points & Technical Q&A:** [`results/experiments/phase16_hardening/JUDGE_QA.md`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/results/experiments/phase16_hardening/JUDGE_QA.md)  
  *(Rigorous answers to 16 key questions distinguishing Measured Fact, Design Decision, Limitation, and Future Work)*
- **System Limitations & Boundary Conditions:** [`results/experiments/phase16_hardening/LIMITATIONS.md`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/results/experiments/phase16_hardening/LIMITATIONS.md)  
  *(Transparent disclosure of ground truth limits, optical occlusion, and computational geometry bottlenecks)*
- **Quick-Start Presentation Guide:** [`DEMO_SETUP.md`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/DEMO_SETUP.md)

---

## 12. Verification of Reference Data Protection

Before and after all Phase 16 hardening tests, the reference prototype directory [`results/prototype/api/8b8c724c7e/`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/results/prototype/api/8b8c724c7e) was cryptographically audited:
- Number of files: 15 files
- File modification times: **100% IDENTICAL before and after all tests**
- Zero accidental writes, zero regressions.

---

## 13. Remaining Blockers & Final Status

- **Known Blockers:** NONE.
- **Regression Tests:** 26 of 26 unit and integration tests passing (`tests/test_api.py`, `tests/test_parcel_reviews.py`, `tests/test_pipeline.py`).
- **Frontend Build:** Vite production bundle builds cleanly in 7.89s with 0 errors.
- **System Health Check:** 10 of 10 checks PASS (`src/health_check.py`).

```
================================================================================
  SIH DEMO STATUS: READY
================================================================================
```
