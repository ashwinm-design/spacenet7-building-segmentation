# SIH26012: High-Resolution Aerial & Drone Building Footprint Segmentation and Cadastral Feature Extraction

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3-61dafb.svg)](https://react.dev/)
[![MapLibre GL](https://img.shields.io/badge/MapLibre_GL-4.7-396afc.svg)](https://maplibre.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-ready AI and geospatial platform for extracting building footprints, mapping road networks, and inferring preliminary cadastral parcel boundaries with human-in-the-loop WebGIS verification.

---

## 🏛️ System Architecture

![SIH26012 System Architecture](results/experiments/phase16_hardening/final_architecture.png)

```
AERIAL / DRONE ORTHOMOSAIC (GeoTIFF)
             ↓
    VALIDATION & TILING (256x256)
             ↓
┌───────────────────────────────────────────────┐
│              DUAL AI EXTRACTION               │
│  - 3-Class Building CNN (Body/Boundary/Back)  │
│  - Dual-Path Roads (Classical CV + DL U-Net)  │
└───────────────────────────────────────────────┘
             ↓
    PHASE 13D HYBRID ROAD FUSION (Topological Graph)
             ↓
    ROAD-CONSTRAINED VORONOI PARCEL ENGINE (100% Valid)
             ↓
    PARCEL INTELLIGENCE (4-Component Confidence & Flags)
             ↓
    HUMAN-IN-THE-LOOP WEBGIS (Live Vertex Dragging & Audit)
             ↓
    STANDARDIZED GIS EXPORTS (GeoJSON, CSV, JSON)
```

---

## 🌟 Key Highlights & Innovations

1. **3-Class Target Formulation:**  
   Unlike naive binary masks that fuse adjacent houses into single blobs, our ResNet34 U-Net predicts **Background, Building Body, and Building Boundary**, maintaining separation across dense informal settlements.
2. **Dual-Path Road Extraction & Hybrid Fusion:**  
   Combines classical morphological computer vision with deep learning semantic segmentation into a topological corridor graph, recovering over 1,570 meters of continuous access networks.
3. **Road-Constrained Preliminary Parcel Engine:**  
   Seeds Voronoi spatial partitioning from building centroids, clipping against road corridors and validating via Shapely `make_valid` to achieve **100.0% topological validity** with zero road encroachment.
4. **Parcel Intelligence & Uncertainty Quantification:**  
   Evaluates each parcel candidate across 4 spatial dimensions (building evidence, road frontage, compactness, and topology) with heuristic uncertainty flags (`WEAK_BUILDING_EVIDENCE`, `MULTI_SOURCE_CONFLICT`).
5. **Human-in-the-Loop WebGIS Verification:**  
   React + MapLibre GL frontend allows surveyors to inspect candidates, drag vertices interactively, update review statuses, and export GIS-ready datasets with an immutable change audit log.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.14.6)
- Node.js v18+ (tested on Node v24.13.0 with npm 11.6.2)

### 2. Backend Setup
```bash
# Clone the repository
git clone https://github.com/your-username/sih26012-cadastral-ai.git
cd sih26012-cadastral-ai

# Install Python dependencies
pip install -r requirements-demo.txt

# Start FastAPI Backend Service (Terminal 1)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
- API Health Check: `http://127.0.0.1:8000/api/health`
- Interactive API Documentation: `http://127.0.0.1:8000/docs`

### 3. Frontend WebGIS Setup
```bash
# In a second terminal (Terminal 2)
cd frontend
npm install
npm run dev
```
- WebGIS Application: `http://localhost:5173`

### 4. Verify System Health
```bash
python src/health_check.py
```

### 5. Automated Terminal Demo (Headless)
```bash
python src/run_demo.py
```

---

## 📊 Benchmark Performance (CPU Hardware)

| Subregion Tile | Resolution | Buildings | Road Length | Parcels | Geometry Validity | Total Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Demo (`f49f31`)** | 1024 × 1024 px | 254 | 1,571.88 m | 254 | **100.0%** | **~38.8 s** |
| **Generalization (`abe1a3`)** | 1024 × 1024 px | 246 | 2,063.66 m | 246 | **100.0%** | **~47.4 s** |

---

## 📦 GIS Deliverables & Export Endpoints

All deliverables are exported in standard open GIS formats:
- **Building Footprints:** GeoJSON (`/api/results/{id}/geojson`)
- **Hybrid Road Corridors:** GeoJSON (`/api/results/{id}/roads?source=hybrid`)
- **Preliminary Parcels:** GeoJSON (`/api/results/{id}/parcels?source=hybrid`)
- **Reviewed Parcels:** GeoJSON (`/api/results/{id}/parcel-reviews/export`)
- **Building Statistics:** CSV (`/api/results/{id}/statistics`)
- **Analysis Summary:** JSON (`/api/results/{id}/summary`)
- **Review Audit Trail:** JSON (`/api/results/{id}/parcel-reviews/audit`)

---

## ⚖️ Legal & Technical Disclaimer

> **IMPORTANT NOTICE:**  
> AI-generated parcel boundaries are preliminary spatial candidates intended for urban planning, spatial reconnaissance, and surveyor assistance. They do **not** establish legal property boundaries, land titles, or registered cadastral rights. Professional ground verification by a licensed surveyor is mandatory prior to official administrative use.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
