# Phase 14B — Human-in-the-Loop WebGIS Parcel Verification

## SIH26012 — AI-Assisted Preliminary Cadastral Spatial Verification Report

> [!IMPORTANT]
> **Legal and Scientific Disclaimer:**
> AI-generated parcel boundaries are preliminary spatial hypotheses and require professional cadastral/survey verification. Review status does not establish legal ownership, cadastral boundaries, land titles, or property rights.

---

## 1. Existing Architecture Audited

Before introducing any modifications, the existing end-to-end system was audited:

- **Backend Architecture (`backend/`):**
  - **Framework:** FastAPI with asynchronous endpoint routing and Pydantic schemas.
  - **Storage:** Disk-based structured directory layout under `results/prototype/api/{analysis_id}/` storing GeoTIFF inputs, polygonized building footprints (`buildings.geojson`), road corridors (`roads.geojson`, `roads_deep_learning.geojson`, `roads_hybrid.geojson`), and candidate parcels (`candidate_parcels.geojson`, `candidate_parcels_dl_constrained.geojson`, `candidate_parcels_hybrid_constrained.geojson`).
  - **Analysis 8b8c724c7e:** Locked reference demo analysis containing 254 building instances in UTM zone 32S (`EPSG:32732`), 4 hybrid road corridors, and 254 preliminary hybrid-constrained parcel candidates.
  - **Phase 14A Audit Results:** Pre-computed in `results/experiments/parcel_intelligence/` (`parcels_audited.geojson`, `audit_summary.json`, `parcel_explanations.json`), containing mean confidence 0.8814, 4-factor evidence breakdowns, and uncertainty flags.

- **Frontend Architecture (`frontend/`):**
  - **Framework:** React 18 + TypeScript + Vite + Tailwind CSS.
  - **Mapping Library:** MapLibre GL JS (WebGL-based vector and raster map rendering) using Carto Dark Matter base tiles.
  - **Reprojection:** Client-side reprojection engine in `frontend/src/utils/geojson.ts` using `proj4` converting source CRS (`EPSG:32732`) to WGS84 (`EPSG:4326`) for web mapping without mutating server deliverables.
  - **Existing Components:** Multi-layer map views (`MapView.tsx`), top navigation (`Header.tsx`), contextual inspector sidebar (`Sidebar.tsx`, `BuildingPanel.tsx`, `AnalysisStats.tsx`, `ExportPanel.tsx`).

---

## 2. Review Data Model

A robust, auditable review state data model was created for every parcel candidate:

| Field | Type | Description |
|---|---|---|
| `candidate_id` | `int` | Sequential parcel candidate identifier (1 to 254) |
| `analysis_id` | `str` | Associated analysis job identifier (e.g. `8b8c724c7e`) |
| `review_status` | `str` | Status: `"unreviewed"`, `"accepted"`, `"edited"`, `"rejected"`, `"needs_review"` (default: `"unreviewed"`) |
| `reviewer_note` | `str \| null` | Human reviewer notes and observations |
| `confidence` | `float` | AI confidence score [0.0 - 1.0] from Phase 14A |
| `confidence_band` | `str` | Confidence band (`"HIGH"`, `"MODERATE"`, `"LOW"`, `"VERY_LOW"`) |
| `flags` | `str \| list[str]` | Uncertainty flags raised during automated auditing |
| `original_geometry` | `dict` | Frozen original AI-generated GeoJSON geometry (Polygon / MultiPolygon) |
| `current_geometry` | `dict` | Current GeoJSON geometry (human-edited or identical to original) |
| `geometry_modified` | `bool` | True if geometry was modified by human reviewer |
| `review_timestamp` | `str \| null` | ISO-8601 UTC timestamp of last review action |
| `area_m2` | `float` | Current parcel footprint area in square meters |
| `building_coverage` | `float` | Building footprint area to parcel area ratio |
| `compactness` | `float` | Polsby-Popper compactness metric [0.0 - 1.0] |
| `road_frontage` | `float` | Road corridor frontage length in meters |
| `nearest_road_distance`| `float` | Distance to nearest physical road corridor |

**Immutability Principle:** `original_geometry` is strictly preserved and never overwritten, enabling seamless rollbacks (`reset`) and complete auditability.

---

## 3. Backend Changes (`backend/`)

1. **Review Storage Manager (`backend/storage/reviews.py`):**
   - Implements `ReviewStorageManager` persisting reviews in `results/prototype/api/{analysis_id}/parcel_reviews.json`.
   - Maintains an append-only audit trail in `results/prototype/api/{analysis_id}/parcel_reviews_audit.json`.
   - Validates geometry using `shapely.geometry.shape`, checks validity, and safely repairs invalid inputs using `make_valid`.
   - Supports both projected coordinates (`EPSG:32732`) and WGS84 (`EPSG:4326`) with automatic pyproj transformation.
   - Recalculates `area_m2`, `building_coverage`, `compactness`, `road_frontage`, and `nearest_road_distance` dynamically upon editing.
   - Generates GIS-ready `reviewed_parcels.geojson` and `review_summary.json`.

2. **Pydantic Schemas (`backend/schemas.py`):**
   - Added `ParcelReviewItem`, `ParcelReviewUpdateRequest`, `ParcelReviewSummary`, and `ParcelReviewListResponse`.

3. **REST Endpoints (`backend/main.py`):**
   - `GET /api/results/{analysis_id}/parcel-reviews`: Returns full review list and aggregate summary.
   - `GET /api/results/{analysis_id}/parcel-reviews/summary`: Returns status breakdown counts.
   - `GET /api/results/{analysis_id}/parcel-reviews/{candidate_id}`: Returns review state and metrics for a single candidate.
   - `POST /api/results/{analysis_id}/parcel-reviews/{candidate_id}`: Accepts status, note, and optional edited geometry.
   - `PATCH /api/results/{analysis_id}/parcel-reviews/{candidate_id}`: RESTful PATCH support for review updates.
   - `POST /api/results/{analysis_id}/parcel-reviews/{candidate_id}/reset`: Resets edited geometry back to original AI hypothesis.
   - `GET /api/results/{analysis_id}/parcel-reviews/export`: Downloads GIS-ready GeoJSON deliverables.
   - Updated `GET /api/results/{analysis_id}/parcels`: Dynamically serves current review geometries and statuses when `source="hybrid"`.

---

## 4. Frontend Changes (`frontend/`)

1. **Type Definitions (`frontend/src/types/analysis.ts`):**
   - Added `ReviewStatus`, `ConfidenceBand`, `ParcelReviewItem`, `ParcelReviewSummary`, `ParcelReviewUpdateRequest`, `ParcelFilterMode`.
   - Augmented `ParcelProperties` with review fields for layer interoperability.

2. **API Client (`frontend/src/api/analysis.ts`):**
   - Added `getParcelReviews()`, `getParcelReviewSummary()`, `getParcelReview()`, `updateParcelReview()`, `resetParcelGeometry()`, `getParcelReviewExportUrl()`.

3. **Parcel Inspection Panel (`frontend/src/components/ParcelInspectionPanel.tsx`):**
   - Dedicated review sidebar displaying Candidate ID, Review Status badge, Confidence %, and Band.
   - Visual Evidence dimension bars: Building Evidence, Road Access, Geometry Quality, Topology Quality.
   - Real-time Spatial Metrics: Area, Building Coverage, Road Frontage, Nearest Road.
   - Uncertainty flags tagged with warning icons.
   - Action buttons: `ACCEPT`, `EDIT GEOM`, `NEEDS REVIEW`, `REJECT`.
   - Reviewer note textarea with save action.
   - `[Show Current]` vs `[Show Original AI]` comparison toggle and `[Reset to Original]` button.
   - Mandatory legal disclaimer notice.

4. **Review Queue Summary Panel (`frontend/src/components/ReviewQueuePanel.tsx`):**
   - Live breakdown table: Total, Unreviewed, Needs Review, Accepted, Edited, Rejected.
   - `[Review Next Candidate]` button using multi-priority sorting.
   - Direct download link for `reviewed_parcels.geojson`.

5. **MapLibre Interactive Map View (`frontend/src/components/MapView.tsx`):**
   - **Review Status Visualization:** Polygons styled distinctly by status (`#00FFA3` unreviewed, `#10B981` accepted, `#3B82F6` edited, `#F59E0B` needs review, `#EF4444` rejected).
   - **Interactive Vertex Editing:** Reviewers can drag polygon vertices, click midpoints to insert vertices, and Alt+click vertices to remove them with live on-map feedback.
   - **Original AI Ghost Outline:** Visual comparison overlay showing the original AI polygon in dashed amber alongside human edits.
   - **Layer Filter Selector:** Filter parcel candidates by review status, confidence band (≥75% vs <75%), or uncertainty flags.
   - **Selection Ring:** Highlight ring (`#FACC15`, width 3.5) around currently selected candidate.

6. **Master Application Integration (`frontend/src/App.tsx`, `Sidebar.tsx`):**
   - Synchronized review state, candidate selection, auto-loading of analysis `8b8c724c7e`, and priority next-review scheduling.

---

## 5. Review Next Prioritization Algorithm

The `Review Next` action implements the deterministic review triage algorithm:

1. **Priority 1:** Candidates with `review_status = "needs_review"` (highest triage priority).
2. **Priority 2:** Candidates with `review_status = "unreviewed"` (pending review).
3. **Secondary Sort:** Lowest confidence first (surfacing highest AI uncertainty, e.g. P56 with 62.54%).
4. **Tie-Breaker:** Deterministic candidate ID ordering (`candidate_id` ascending).

---

## 6. Geometry Editing & Attribute Recalculation

When a reviewer modifies a parcel polygon:
1. Coordinates are captured in WGS84 and sent to the backend.
2. The backend reprojects coordinates to `EPSG:32732` (UTM zone 32S).
3. Geometry is validated using Shapely (`is_valid`, `make_valid` repair if needed).
4. Geometry-dependent metrics are recomputed:
   - $\text{Area} = \text{geom.area}$
   - $\text{Polsby-Popper Compactness} = \frac{4\pi \times \text{Area}}{\text{Perimeter}^2}$
   - $\text{Building Coverage} = \frac{\text{Building Footprint Area}}{\text{Parcel Area}}$
   - $\text{Road Distance} = \text{geom.distance}(\text{Road Corridor Union})$
   - $\text{Road Frontage} = \text{Length}(\partial \text{Parcel} \cap \text{Road Corridor Buffer})$
5. Stale values from the original AI geometry are replaced.
6. The action is recorded in the audit trail with SHA-256 hashes of the old and new geometries.

---

## 7. Automated Test Suite Validation

Ten automated tests in `tests/test_parcel_reviews.py` were executed and passed (10/10):

| Test ID | Test Description | Result |
|---|---|---|
| `test_get_parcel_reviews_list` | GET `/api/results/{id}/parcel-reviews` list and queue summary | **PASSED** |
| `test_get_individual_parcel_p56` | GET `/api/results/{id}/parcel-reviews/56` low-confidence candidate | **PASSED** |
| `test_accept_parcel` | POST status=`"accepted"` with reviewer note | **PASSED** |
| `test_mark_needs_review` | POST status=`"needs_review"` with reviewer note | **PASSED** |
| `test_reject_parcel` | PATCH status=`"rejected"` | **PASSED** |
| `test_edit_valid_geometry` | Edit geometry and verify recalculation of area, coverage, compactness | **PASSED** |
| `test_reject_invalid_geometry` | Reject NaN coordinates, non-polygon types, and zero-area degenerates (400/422) | **PASSED** |
| `test_reset_edited_geometry` | POST `/reset` restores original AI geometry and reverts metrics | **PASSED** |
| `test_export_reviewed_geojson` | GET `/export` downloads valid GeoJSON FeatureCollection with all properties | **PASSED** |
| `test_review_persistence_and_audit` | Verify persistent JSON store and append-only audit trail logging | **PASSED** |

Existing API tests in `tests/test_api.py` were also executed and passed completely (6/6).
Frontend production build (`npm run build`) succeeded with 0 errors.

---

## 8. Demo Workflow Results (Analysis `8b8c724c7e`)

The automated demo script `src/demo_parcel_review.py` was executed on analysis `8b8c724c7e`:

1. **Load WebGIS:** Successfully loaded analysis metadata (`EPSG:32732`, 254 buildings).
2. **Hybrid Roads:** Loaded 4 fused road corridor features.
3. **Display Parcels:** Loaded 254 preliminary hybrid parcel candidates.
4. **Inspect P056:** Confidence 62.54% (MODERATE), Building Evidence 0.1535, Road Access 0.5175, Geometry 1.0, Topology 1.0, Area 61.0 m², Flags: `WEAK_BUILDING_EVIDENCE,MULTI_SOURCE_CONFLICT`.
5. **Mark Needs Review:** Status changed to `NEEDS_REVIEW` with note: *"Boundary conflicts with neighbor footprint; recommended field cadastral survey."*
6. **Edit Geometry (P012):** Adjusted boundary vertices. Area recalculated from 22.47 m² to 30.13 m²; compactness updated to 0.6652; building coverage updated to 5.44%; `geometry_modified = True`; status changed to `EDITED`.
7. **Accept Candidate (P006):** High confidence (97.60% HIGH) candidate accepted with note: *"Boundary aligns cleanly with road frontage (13.6m) and structure footprints."*
8. **Review Queue State:**
   - Total Candidates: 254
   - Unreviewed: 250
   - Accepted: 1
   - Edited: 1
   - Needs Review: 1
   - Rejected: 1
9. **GIS Export Validation:** Verified `reviewed_parcels.geojson` (254 features) and `review_summary.json`.

---

## 9. Deliverables & Files Summary

### Files Created
- [`backend/storage/reviews.py`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/backend/storage/reviews.py): Review data model, storage manager, geometry validation, recalculation engine, audit trail, and export handler.
- [`frontend/src/components/ParcelInspectionPanel.tsx`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/frontend/src/components/ParcelInspectionPanel.tsx): Detailed parcel intelligence and review action card.
- [`frontend/src/components/ReviewQueuePanel.tsx`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/frontend/src/components/ReviewQueuePanel.tsx): Review queue breakdown and Review Next control.
- [`tests/test_parcel_reviews.py`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/tests/test_parcel_reviews.py): 10 backend validation tests for all review endpoints and operations.
- [`src/demo_parcel_review.py`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/src/demo_parcel_review.py): End-to-end demonstration script.
- [`results/experiments/parcel_review/reviewed_parcels.geojson`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/results/experiments/parcel_review/reviewed_parcels.geojson): GIS-ready GeoJSON export containing all reviewed parcel candidates.
- [`results/experiments/parcel_review/review_summary.json`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/results/experiments/parcel_review/review_summary.json): Current review queue status breakdown.
- [`results/experiments/parcel_review/report.md`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/results/experiments/parcel_review/report.md): This report.

### Files Modified
- [`backend/schemas.py`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/backend/schemas.py): Added Pydantic review schemas.
- [`backend/main.py`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/backend/main.py): Added review endpoints and dynamic parcel layer serving.
- [`frontend/src/types/analysis.ts`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/frontend/src/types/analysis.ts): Added TypeScript interfaces for reviews, statuses, and filters.
- [`frontend/src/api/analysis.ts`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/frontend/src/api/analysis.ts): Added client service functions for reviews and exports.
- [`frontend/src/utils/geojson.ts`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/frontend/src/utils/geojson.ts): Added `reprojectGeometryForDisplay` helper for original AI ghost overlay.
- [`frontend/src/components/Sidebar.tsx`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/frontend/src/components/Sidebar.tsx): Integrated inspection and review queue panels.
- [`frontend/src/components/MapView.tsx`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/frontend/src/components/MapView.tsx): Implemented status coloring, vertex dragging, ghost outline, and filtering.
- [`frontend/src/App.tsx`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/frontend/src/App.tsx): Wired up review state, actions, editing, and Review Next logic.

---

## 10. Commands Executed

```bash
# Verify packages and inspect existing analysis metadata
python -c "from backend.storage.reviews import review_storage; print(review_storage.get_review('8b8c724c7e', 56))"
python -c "from backend.services.result_service import get_analysis_metadata; print(get_analysis_metadata('8b8c724c7e'))"

# Run automated backend test suite
python -m pytest tests/test_parcel_reviews.py
python -m pytest tests/test_api.py

# Build frontend application bundle
cd frontend && npm run build

# Run demo workflow script
python src/demo_parcel_review.py
```

---

## 11. Limitations & Scope

1. **Prototype WebGIS Environment:** Geometries edited in Web Mercator projection are converted to the projected UTM CRS (`EPSG:32732`) with sub-meter numerical precision. However, this tool serves spatial decision support and preliminary mapping; it is not a sub-centimeter geodetic survey CAD platform.
2. **Topological Inter-Parcel Gaps:** Individual parcel polygon vertex dragging adjusts the boundary of the selected parcel candidate. In this phase, adjacent shared boundaries do not automatically perform topology-locked rubber-sheeting (which would belong to an advanced multi-parcel cadastral topological engine).
3. **Legal Disclaimer Strictly Maintained:** Review acceptance indicates approval by the human analyst for map hypothesis refinement and does not certify legal cadastral title or property rights.
