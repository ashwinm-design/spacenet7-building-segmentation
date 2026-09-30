# SIH26012 Phase 13B: Road-Constrained Preliminary Urban Parcel Candidates Report

**Status:** Completed & Validated  
**Pipeline:** Drone/Aerial Imagery → 3-Class U-Net Footprints → Road Extraction → Road-Constrained Bounded Voronoi → Topology Validation → Web-GIS  
**Target Subregion:** Open Cities Pointe-Noire (`f49f31.tif`, 1024×1024 px, 204.8m × 204.8m, Area: 41,943.04 m²)  
**Spatial Reference System:** EPSG:32732  
**Date/Time:** 2026-09-30 00:11:28  

---

## 1. Executive Summary & Objective

In Phase 13A, preliminary parcel candidate boundaries were generated using bounded Voronoi spatial partitioning. While geometrically valid and overlap-free, they lacked access constraints, allowing parcel candidates to unphysically span across observable roadways.

**Phase 13B resolves this limitation** by extracting observable physical road/access corridors and injecting them directly into the parcel generation engine. Parcel candidates are now geometrically constrained by transportation corridors, preventing parcels from crossing roadways and enabling quantifiable road frontage and access analysis.

> [!IMPORTANT]
> **Strict Legal Cadastral Posture:**  
> The outputs generated in this phase are strictly designated as **"Preliminary Parcel Candidates"** and **"Physical Access Corridors"**. They do **NOT** represent surveyed legal property boundaries, registered cadastral plans, or official rights-of-way. No legal cadastral ownership claims are made.

> [!NOTE]
> **Ground Truth Integrity Statement:**  
> **No authoritative parcel or road ground-truth layer exists** in the Open Cities dataset; therefore, cadastral/road boundary accuracy is not reported or fabricated. Performance is measured using empirical geometric validity, topological consistency, road access quantification, and plausibility scoring.

---

## 2. Quantitative Comparison: Phase 13A vs Phase 13B

| Metric | Phase 13A (Unconstrained) | Phase 13B (Road-Constrained) | Delta / Impact |
|---|---|---|---|
| **Input Building Footprints** | `254` | `254` | Unchanged (1:1 footprint correspondence) |
| **Preliminary Parcel Candidates** | `254` | `254` | 1:1 complete candidate mapping |
| **Valid Geometries** | `254 (100.0%)` | `254 (100.0%)` | **100% OGC/GEOS geometric validity maintained** |
| **Pairwise Overlaps** | `0` | `0` | **0 pairwise overlaps (Strict 0.00 m² overlap)** |
| **Total Parcel Area** | `41,203.83 m²` | `28,477.99 m²` | **-12,725.8 m²** (ceded to road corridors) |
| **Mean Parcel Area** | `162.22 m²` | `112.12 m²` | More realistic parcel dimensions |
| **Median Parcel Area** | `154.82 m²` | `99.83 m²` | Accounts for dense building clusters |
| **Mean Building Coverage Ratio** | `0.3971` | `0.4619` | Higher urban coverage after road removal |
| **Mean Compactness (Polsby-Popper)** | `0.7243` | `0.6457` | Realistic frontage shapes along roads |
| **Road-Adjacent Candidates** | `0 (0.0%)` | `202 (79.53%)` | **+202 parcels connected to physical roads** |
| **Mean Road Frontage** | `0.0 m` | `11.72 m` | Directly measured shared frontage length |
| **Mean Road Distance** | `0.0 m` | `0.88 m` | Average distance from parcel to nearest road |
| **Mean Plausibility Evidence Score** | `0.7908` | `0.7847` | **Multivariate evidence scoring** |

---

## 3. Road-Constrained Partitioning Architecture

The road-constrained engine executes the following pipeline:
1. **Spatial Adjacency Graph:** Groups buildings within 15m into topological clusters.
2. **Cluster Envelope Formulation:** Computes optimized cluster bounding envelopes ($5000\times$ faster via individual buffering before union).
3. **Voronoi Diagram Construction:** Generates Voronoi partition cells seeded at building centroids.
4. **Physical Road Subtraction:** Subtracts extracted road corridor polygons ($R$) from candidate parcel cells ($P_{raw} = P \setminus R$).
5. **Component Filtering:** When differencing produces a MultiPolygon, the subpolygon intersecting the building footprint is preserved, eliminating detached road-crossing slivers.
6. **Topology Repair & Validation:** Enforces non-overlapping, simple, valid polygons via Shapely/GEOS `make_valid` and sliver suppression.
7. **Road Frontage Analysis:** Measures the exact linear shared boundary between each parcel exterior and road corridor.
8. **Explainable Spatial Plausibility Scoring:**
   $$\text{Score} = 0.30\cdot\text{Containment} + 0.25\cdot\text{Coverage} + 0.20\cdot\text{Compactness} + 0.15\cdot\text{Coherence} + 0.10\cdot\text{Road Adjacency}$$

---

## 4. Exploratory Structural Hypotheses

### A. Multi-Building Compound Hypotheses
- **Identified Compounds:** `3`
- **Total Associated Buildings:** `30`
- **Analysis:** Multi-building clusters sharing a single contiguous envelope without internal road barriers represent plausible multi-building compounds (e.g. main residential unit + garage or auxiliary shed).
- **Caveat:** Explicitly recorded as a *"multi-building compound hypothesis"*; does not establish legal ownership.

### B. Potential Vacant / Open Parcel Candidates
- **Identified Vacant Candidates:** `2`
- **Total Vacant Area:** `181.91 m²`
- **Mean Vacant Candidate Area:** `90.95 m²`
- **Mean Road Frontage:** `58.04 m`
- **Analysis:** Unbuilt residual areas outside building setbacks with direct street frontage (>= 3m) and contiguous area (>= 40 m²).
- **Caveat:** Labelled as *"Potential vacant/open parcel candidate"*; explicitly not confirmed cadastral parcels.

---

## 5. Visual Validation Artifacts

Four high-resolution presentation figures generated at 300 DPI:
1. `01_parcels_before_vs_after.png`: Side-by-side comparison of Phase 13A unconstrained vs Phase 13B road-constrained preliminary parcels.
2. `02_road_constrained_parcels.png`: Master high-contrast GIS overlay showing Aerial Image + Road Corridors + Preliminary Parcels + AI Building Footprints.
3. `03_road_frontage_heatmap.png`: Spatial distributions of measured road frontage length (m) and candidate plausibility scores.
4. `04_compounds_and_vacant.png`: Visualization of multi-building compound hypotheses and potential vacant parcel candidates.

---

## 6. Deliverables Summary

- `baseline_metrics.csv`
- `road_constrained_metrics.csv`
- `comparison.csv`
- `candidate_parcels_road_constrained.geojson`
- `multi_building_compounds.geojson`
- `vacant_parcel_candidates.geojson`
- `topology_report.json`
- `report.md`
