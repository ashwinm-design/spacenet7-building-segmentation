# Environment Audit — SIH26012 Demo Hardening

**Date:** September 30, 2026  
**Project:** SIH26012 — High-Resolution Aerial & Drone Building Footprint Segmentation & Cadastral Feature Extraction  
**Phase:** Phase 16 — Demo Hardening & Presentation Readiness  

---

## 1. Operating System & Host Environment

- **Operating System:** Windows 11 Home / Pro (x64-based PC)
- **Kernel / OS Version:** `Windows-11-10.0.26200-SP0`
- **Shell:** PowerShell 7 / Windows PowerShell / Command Prompt
- **Hardware Profile:**
  - Architecture: AMD64 / x86_64
  - Acceleration: CPU Execution (Multi-threaded PyTorch + NumPy + OpenCV + Shapely)
  - PyTorch CUDA Status: `False` (CPU Mode verified and benchmarked)

---

## 2. Runtimes & Interpreters

| Runtime | Measured Version | Role in SIH26012 System |
| :--- | :--- | :--- |
| **Python** | `3.14.6 (tags/v3.14.6:c63aec6, Jun 10 2026)` | Backend REST API, ML Inference, CV, Spatial Geometry |
| **Node.js** | `v24.13.0` | Frontend Development & Production Build Tooling |
| **npm** | `11.6.2` | Package management for Vite/React WebGIS frontend |

---

## 3. Python Package Stack (Measured Versions)

| Package | Installed Version | Primary Role |
| :--- | :--- | :--- |
| `torch` | `2.14.0` | Deep learning model inference (3-class building & road DL) |
| `torchvision` | `0.29.0` | Neural network backbone layers & image transforms |
| `rasterio` | `1.5.1` | GeoTIFF raster reading, georeferencing, transform math |
| `geopandas` | `1.2.0` | GeoDataFrame vector operations, spatial join, GeoJSON I/O |
| `shapely` | `2.1.2` | 2D computational geometry, Polygon/MultiPolygon validation |
| `pyproj` | `3.8.0` | Coordinate Reference System (CRS) transformations |
| `pyogrio` | `0.13.0` | High-performance vector I/O engine for GeoPandas |
| `numpy` | `2.5.1` | Array computing, matrix transformations, masking |
| `scipy` | `1.18.0` | Voronoi tessellation, Delaunay triangulation, distance transforms |
| `opencv-python` | `5.0.0.93` | Classical CV filtering, morphology, skeletonization |
| `fastapi` | `0.141.1` | Async REST API framework |
| `uvicorn` | `0.54.0` | ASGI production server for FastAPI |
| `pydantic` | `2.13.5` | Data validation, contract typing, settings management |
| `requests` | `2.34.2` | HTTP client for external integrations & health checks |
| `httpx` | `0.28.1` | Async HTTP client for FastAPI TestClient test suite |
| `scikit-learn` | `1.9.1` | Spatial clustering, DBSCAN road graph simplification |
| `matplotlib` | `3.11.1` | Visualization overlay rendering and static figure generation |
| `pillow` | `12.3.0` | Image format conversion, PNG visualization encoding |
| `networkx` | `3.7` | Dual-layer road topological graph representation |

---

## 4. Frontend Technology Stack (Vite + React)

- **Framework:** React 18.3.1 (`react`, `react-dom`)
- **Language:** TypeScript 5.6.3
- **Build Engine:** Vite 5.4.11 / 5.4.21
- **Styling:** Tailwind CSS 3.4.16 + Autoprefixer 10.4.20
- **WebGIS Engine:** MapLibre GL 4.7.1 (`maplibre-gl`)
- **Spatial Vector Editor:** `@mapbox/mapbox-gl-draw` 1.5.2
- **Coordinate Projection:** `proj4` 2.15.0
- **HTTP Client:** `axios` 1.7.9
- **Iconography:** `lucide-react` 0.468.0

---

## 5. Network Ports & Communication Topology

| Service | Host | Port | Protocol | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Backend REST API** | `127.0.0.1` | `8000` | HTTP | FastAPI server exposing inference, status, reviews, and exports |
| **Frontend WebGIS** | `localhost` | `5173` | HTTP | Vite dev server / production preview hosting WebGIS UI |

- **CORS Allowed Origins:** `http://localhost:5173`, `http://127.0.0.1:5173`, `http://localhost:3000`, `http://localhost:8000`, `*`.
- **Proxying:** Vite dev server automatically proxies `/api` calls to `http://localhost:8000`.

---

## 6. Environment Variables

| Variable | Default Value | Required? | Description |
| :--- | :--- | :--- | :--- |
| `VITE_API_BASE_URL` | `http://localhost:8000` | Optional | Specifies the FastAPI backend URL for Axios client requests |
| `PYTHONUNBUFFERED` | `1` | Recommended | Ensures immediate stdout flushing for pipeline progress logging |
| `PYTHONIOENCODING` | `utf-8` | Recommended | Prevents CP1252 character mapping errors on Windows terminal consoles |

---

## 7. Model Checkpoints & Storage Locations

| Artifact | Relative Path | Size | Status |
| :--- | :--- | :--- | :--- |
| **3-Class Building Model** | `results/checkpoints/best_model.pth` | 164.15 MB | **Frozen** (SpaceNet 7 ResNet34 U-Net) |
| **Road Deep Learning Model** | `results/experiments/road_dl/checkpoints/best_road_model.pth` | 54.76 MB | **Frozen** (ResNet18 U-Net) |
| **Prototype Reference Run** | `results/prototype/api/8b8c724c7e/` | ~19 MB | **Locked** (Untouched benchmark baseline) |

---

## 8. Required Startup Sequence

1. **Verify Python & Node environments:**
   ```bash
   python -c "import torch, rasterio, geopandas, fastapi; print('Python dependencies OK')"
   node -v
   ```
2. **Start Backend Service (Terminal 1):**
   ```bash
   python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```
3. **Verify Health Endpoint:**
   `curl http://127.0.0.1:8000/api/health` $\rightarrow$ `{"status":"ok","service":"SIH26012 AI Mapping Backend"}`
4. **Start Frontend Service (Terminal 2):**
   ```bash
   cd frontend
   npm run dev
   ```
5. **Open Browser:** Navigate to `http://localhost:5173`.
