# Phase 15 — End-to-End Integration & SIH Demo Readiness Report

**Project:** SIH26012 — High-Resolution Aerial & Drone Building Footprint Segmentation and Cadastral Feature Extraction  
**Date:** September 30, 2026  
**Status:** Completed & Validated  

---

## 1. Executive Summary

Phase 15 accomplishes the full end-to-end integration and reliability verification of the SIH26012 system. Rather than introducing new machine learning architectures or retraining existing models, this phase unified all previously established subsystems into a seamless, observable, and production-grade pipeline:

1. **Building Footprint AI (Phases 1–12):** 3-Class CNN segmentation (Body, Boundary, Background) trained on SpaceNet 7 and adapted zero-shot to Open Cities imagery.
2. **Road Extraction & Hybrid Fusion (Phases 13A–13D):** Morphological classical CV, deep learning U-Net ResNet18, Phase 13D topological fusion, and dual-layer graph representation.
3. **Preliminary Parcel Engine (Phase 13D):** Road-constrained Voronoi spatial partitioning generating topologically plausible preliminary parcel hypotheses.
4. **Parcel Intelligence (Phase 14A):** 4-component weighted confidence scoring and multi-criteria uncertainty flagging.
5. **Human-in-the-Loop Verification (Phase 14B):** Review queue management, interactive status transitions, geometric recalculation, and immutable change auditing.
6. **WebGIS Interface & REST API (Phases 13–15):** React/Vite/Leaflet frontend coupled with FastAPI backend supporting real-time stage progress polling and comprehensive GIS export.

All operations strictly adhered to the non-invasive constraints: no models were retrained, frozen checkpoints remained bit-for-bit identical, reference directory `8b8c724c7e` was preserved untouched, and cadastral boundaries are clearly documented as preliminary spatial hypotheses.

---

## 2. Integrated Pipeline Architecture

The end-to-end orchestrator (`backend/services/pipeline_service.py`) manages 10 sequential, weighted stages with real-time status reporting:

```
[01: VALIDATING] (5%)
       ↓
[02: PREPROCESSING] (10%)
       ↓
[03: BUILDING_INFERENCE] (30%) ───────> 3-Class Segmentation
       ↓
[04: BUILDING_POLYGONIZATION] (40%) ──> Vector Footprints + Statistics CSV + PNG Overlay
       ↓
[05: ROAD_EXTRACTION] (55%) ──────────> Classical CV + Deep Learning Corridors
       ↓
[06: ROAD_FUSION] (65%) ──────────────> Phase 13D Hybrid Network + Topology Graph
       ↓
[07: PARCEL_GENERATION] (75%) ────────> Road-Constrained Voronoi Partitioning
       ↓
[08: PARCEL_INTELLIGENCE] (85%) ──────> Multi-Component Confidence + Uncertainty Flags
       ↓
[09: READY_FOR_REVIEW] (95%) ─────────> Human Review Store & Initialized Audit Log
       ↓
[10: COMPLETED] (100%) ───────────────> Package Deliverables & Dashboard Summary
```

### Stage Weighting & Status Contracts

| Stage | Progress Weight | Operation Description | Primary Output Artifact |
| :--- | :---: | :--- | :--- |
| `QUEUED` | 0% | Job scheduled in memory | `pipeline_status.json` |
| `VALIDATING` | 5% | Raster format, dimensions, channels, NaN/Inf, and CRS integrity | Validated metadata |
| `PREPROCESSING` | 10% | Tiled grid computation (256x256 tiles, 192 stride) and ROI window slicing | RGB numpy tensor |
| `BUILDING_INFERENCE` | 30% | Deep learning segmentation inference on tiles with cosine blending | Class map & probability raster |
| `BUILDING_POLYGONIZATION` | 40% | Contour vectorization, Douglas-Peucker simplification, attribute calculation | `buildings.geojson`, `statistics.csv`, `overlay.png` |
| `ROAD_EXTRACTION` | 55% | Building exclusion, classical morphological filtering, and DL road prediction | `roads.geojson`, `roads_deep_learning.geojson` |
| `ROAD_FUSION` | 65% | Phase 13D topological fusion, graph node/edge creation, DL candidate recovery | `roads_hybrid.geojson`, `road_graph_*.geojson` |
| `PARCEL_GENERATION` | 75% | Voronoi polygon clipping against road corridor buffer mask | `candidate_parcels_hybrid_constrained.geojson` |
| `PARCEL_INTELLIGENCE` | 85% | 4-component confidence scoring, heuristic uncertainty flagger | `parcels_audited.geojson` |
| `READY_FOR_REVIEW` | 95% | Phase 14B review queue initialization, export sync, and audit log init | `parcel_reviews.json`, `reviewed_parcels.geojson` |
| `COMPLETED` | 100% | Final KPI aggregation, timing calculation, and metadata generation | `analysis_summary.json`, `metadata.json` |

---

## 3. Deliverables & Artifact Isolation

Every pipeline execution generates a strictly isolated analysis directory under `results/prototype/api/{analysis_id}/` containing exactly 17 intermediate and final artifacts:

```
results/prototype/api/{analysis_id}/
├── input.tif                                    (Uploaded / demo raster source)
├── pipeline_status.json                         (Real-time progress state)
├── buildings.geojson                            (Vectorized building footprints)
├── statistics.csv                               (Instance-level morphological dimensions)
├── overlay.png                                  (High-resolution visual QA overlay)
├── roads.geojson                                (Classical CV road corridors)
├── roads_deep_learning.geojson                  (U-Net ResNet18 DL road corridors)
├── roads_hybrid.geojson                         (Phase 13D hybrid fused road network)
├── road_graph_nodes.geojson                     (Topological network junctions)
├── road_graph_edges.geojson                     (Topological network segments)
├── candidate_parcels_hybrid_constrained.geojson (Road-constrained raw parcel polygons)
├── parcels_hybrid.geojson                       (Standardized hybrid parcel candidates)
├── parcels_audited.geojson                      (Parcels enriched with 4 confidence scores & flags)
├── parcel_reviews.json                          (Active review state store for human verification)
├── parcel_reviews_audit.json                    (Immutable change log of all human interactions)
├── reviewed_parcels.geojson                     (GIS-ready export with human verification attributes)
├── review_summary.json                          (Current review queue breakdown)
├── analysis_summary.json                        (Complete Phase 15 KPI & timing summary)
└── metadata.json                                (Backward-compatible API summary metadata)
```

---

## 4. Empirical Performance & Stage Timings

Benchmarks were performed on a standard multi-core CPU environment without GPU acceleration:

### Run Comparison

| Metric / Stage | Demo Run (`f49f31`) | Generalization Run (`abe1a3`) |
| :--- | :---: | :---: |
| **Tile Location** | `data/open_cities_raw/ptn/f49f31` | `data/open_cities_raw/ptn/abe1a3` |
| **ROI Window** | `(3072, 2048, 4096, 3072)` | `(3000, 3000, 4024, 4024)` |
| **Subregion Dimensions** | 1024 × 1024 px | 1024 × 1024 px |
| **Coordinate Reference System** | EPSG:32732 (UTM zone 32S) | EPSG:32732 (UTM zone 32S) |
| **Stage 1: Input Validation** | 0.103 s (0.3%) | 0.103 s (0.2%) |
| **Stage 2: Preprocessing** | 0.414 s (1.1%) | 0.953 s (2.0%) |
| **Stage 3: Building Inference** | 1.462 s (3.8%) | 1.015 s (2.1%) |
| **Stage 4: Building Polygonization** | 3.437 s (8.9%) | 2.390 s (5.0%) |
| **Stage 5: Road Extraction** | 3.328 s (8.6%) | 3.628 s (7.6%) |
| **Stage 6: Road Hybrid Fusion** | 6.205 s (16.0%) | 11.217 s (23.6%) |
| **Stage 7: Parcel Generation** | 23.170 s (59.7%) | 27.594 s (58.1%) |
| **Stage 8: Parcel Intelligence** | 0.513 s (1.3%) | 0.447 s (0.9%) |
| **Stage 9: Ready for Review** | 0.170 s (0.4%) | 0.111 s (0.2%) |
| **Total Processing Time** | **38.80 s** | **47.46 s** |

### Bottleneck Analysis
- **Primary Bottleneck (~58–60%):** Stage 7 (`PARCEL_GENERATION`). Spatial difference operations subtracting buffered road networks from Voronoi polygons across 250+ polygons require intensive computational geometry.
- **Secondary Bottleneck (~16–24%):** Stage 6 (`ROAD_FUSION`). Topological path extraction, candidate recovery, and graph junction simplification.
- **Sub-Second Modules:** Preprocessing (0.4–0.9s), Building Inference (1.0–1.5s), Parcel Intelligence scoring (0.4–0.5s), Review Store initialization (0.1–0.2s).

---

## 5. Cross-Tile Generalization & Feature Extraction

The pipeline demonstrated full dynamic adaptation across different geographic tiles without parameter retuning or hardcoded assumptions:

```
                                  Feature Extraction Comparison
┌─────────────────────────────────┬──────────────────────────┬──────────────────────────┐
│ Extraction Feature              │ Demo Tile (f49f31)       │ Generalization (abe1a3)  │
├─────────────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Extracted Buildings             │ 254 footprints           │ 246 footprints           │
│ Hybrid Road Corridors           │ 4 network corridors      │ 8 network corridors      │
│ Total Road Corridor Length      │ 1,571.88 meters          │ 2,063.66 meters          │
│ Generated Parcel Hypotheses     │ 254 candidate parcels    │ 246 candidate parcels    │
│ Parcel Geometry Validity        │ 100.0% (254/254 valid)   │ 100.0% (246/246 valid)   │
│ Confidence Score Mean           │ 88.17%                   │ 89.11%                   │
│ Confidence Score Median         │ 89.53%                   │ 90.10%                   │
│ Confidence Range [Min - Max]    │ [62.54% - 100.0%]        │ [64.66% - 100.0%]        │
│ Flagged for Human Review        │ 210 candidates           │ 224 candidates           │
│ Initial Review Queue Status     │ 254 Unreviewed (100%)    │ 246 Unreviewed (100%)    │
└─────────────────────────────────┴──────────────────────────┴──────────────────────────┘
```

---

## 6. REST API Surface & Frontend Integration

The backend exposes 20 REST endpoints adhering to strict Pydantic schemas:

### Endpoint Catalog

| Method | Endpoint Route | Request / Response Model | Purpose |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | `HealthResponse` | Operational status check |
| `POST` | `/api/analyze` | Form upload $\rightarrow$ `Union[Creation, Response]` | Analyze image with `background=true/false` |
| `POST` | `/api/demo` | `background=true/false` $\rightarrow$ `Response` | Execute default demo subregion |
| `GET` | `/api/analysis/{id}/status` | `PipelineStatusResponse` | Real-time progress %, active stage, error |
| `GET` | `/api/results/{id}/summary` | `AnalysisSummaryData` | Comprehensive KPI and stage timing JSON |
| `GET` | `/api/results/{id}` | `AnalysisResponse` | Legacy summary metadata |
| `GET` | `/api/results/{id}/geojson` | GeoJSON FeatureCollection | Building footprints in source CRS |
| `GET` | `/api/results/{id}/roads` | GeoJSON FeatureCollection | Road corridors (`classical`/`dl`/`hybrid`) |
| `GET` | `/api/results/{id}/parcels` | GeoJSON FeatureCollection | Preliminary parcels (`hybrid`/`reviewed`) |
| `GET` | `/api/results/{id}/statistics` | CSV text/csv | Instance-level footprint metrics |
| `GET` | `/api/results/{id}/overlay` | PNG image/png | Visual overlay render |
| `GET` | `/api/results/{id}/buildings/{bid}` | `BuildingDetailResponse` | Single building geometry & metrics |
| `GET` | `/api/results/{id}/parcel-reviews` | `ParcelReviewListResponse` | Full review queue & summary counts |
| `GET` | `/api/results/{id}/parcel-reviews/summary`| `ParcelReviewSummary` | Review queue category counters |
| `GET` | `/api/results/{id}/parcel-reviews/{cid}` | `ParcelReviewItem` | Candidate metrics, flags, and review status |
| `POST` | `/api/results/{id}/parcel-reviews/{cid}` | `ParcelReviewUpdateRequest` | Accept, flag, edit, or reject parcel |
| `PATCH`| `/api/results/{id}/parcel-reviews/{cid}` | `ParcelReviewUpdateRequest` | PATCH alternative for review updates |
| `POST` | `/api/results/{id}/parcel-reviews/{cid}/reset` | `ParcelReviewItem` | Revert geometry to original AI hypothesis |
| `GET` | `/api/results/{id}/parcel-reviews/export` | GeoJSON FeatureCollection | GIS-ready reviewed parcels export |
| `GET` | `/api/results/{id}/parcel-reviews/audit` | JSON array of audit objects | Immutable history of all review actions |

### Frontend UI Enhancements

1. **ProcessingStatus Component:** Upgraded with animated progress bar, real-time stage checklist (`QUEUED` $\rightarrow$ `COMPLETED`), active analysis ID badge, KPI summary display, and a direct `[ OPEN WEBGIS ]` transition button.
2. **AnalysisStats Component:** Enhanced dashboard displaying building count, total road corridor meters, parcel hypotheses, geometry validity %, and real-time review breakdown badges (Accepted, Edited, Needs Review, Rejected).
3. **ExportPanel Component:** Provides 1-click downloads for all 6 primary GIS deliverables:
   - Building Footprints (`GeoJSON`)
   - Hybrid Road Corridors (`GeoJSON`)
   - Preliminary Parcels (`GeoJSON`)
   - Reviewed Parcels (`GeoJSON`)
   - Analysis Summary (`JSON`)
   - Verification Audit Log (`JSON`)
4. **Interactive Polling Engine:** Frontend switches seamlessly from background polling (`/api/analysis/{id}/status` every 800ms) to GIS visualization once status reaches `completed`.

---

## 7. Verification of Non-Invasive Constraints

All Phase 15 requirements were validated without compromising existing components:

1. **Frozen Checkpoints Unmodified:**
   - SpaceNet 7 Building Model: `results/checkpoints/best_model.pth` (untouched)
   - Road DL U-Net ResNet18: `results/experiments/road_dl/checkpoints/best_road_model.pth` (untouched)
2. **No Model Retraining:** All inference utilizes pre-trained frozen weights.
3. **Reference Analysis Intact:** `results/prototype/api/8b8c724c7e/` files remained bit-for-bit unchanged before, during, and after all test runs.
4. **Automated Test Suite:** 26 automated unit and integration tests passing (`tests/test_api.py`, `tests/test_parcel_reviews.py`, `tests/test_pipeline.py`).
5. **No Fabricated Data:** All metrics represent genuine computer vision and geometric processing outputs.

---

## 8. Cadastral Boundary Disclaimer & Legal Scope

> [!WARNING]
> **LEGAL AND SURVEY DISCLAIMER**
> AI-generated parcel boundaries are preliminary spatial candidates derived from computational geometry (road-constrained Voronoi partitioning and building proximity). They are intended exclusively for spatial planning, rapid cadastral reconnaissance, and human-in-the-loop verification. They do **not** constitute legal property boundaries, land titles, or registered cadastral surveys. Professional field verification by a licensed surveyor is mandatory prior to legal or administrative use.
