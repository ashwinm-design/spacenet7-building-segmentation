# SIH26012 Phase 13B — Physical Road & Access Corridor Extraction Report

**Status:** Completed & Validated  
**Target Area:** Open Cities Drone/Aerial Orthomosaic (`f49f31.tif`, Pointe-Noire)  
**Spatial Reference:** EPSG:32732 (UTM Zone 32S)  
**ROI Subregion:** 1024×1024 px (204.8m × 204.8m | Area: 41,943.04 m²)  
**Timestamp:** 2026-09-30 00:03:15  

---

## 1. Executive Summary

Phase 13B introduces reproducible, GIS-ready physical road and access corridor extraction into the SIH26012 pipeline to constrain preliminary parcel candidate generation.

> [!IMPORTANT]
> **Legal Posture & Terminology:**  
> Road extraction is strictly treated as a **physical-feature extraction problem**, NOT a legal right-of-way determination problem. The extracted corridors represent observable, unpaved and paved physical transportation routes. They do NOT represent officially surveyed legal road boundaries or public right-of-way limits.

> [!NOTE]
> **Ground Truth Integrity Statement:**  
> **No authoritative road ground-truth layer was available** in the Open Cities dataset; therefore road accuracy (IoU, precision, recall) is NOT reported or fabricated. Quality is evaluated via geometric validity, continuity, width estimates, and structural non-encroachment on building footprints.

---

## 2. Quantitative Road Extraction Metrics

| Metric | Measured Value | Unit / Note |
|---|---|---|
| **Road Component Count** | `6` | Distinct physical corridor corridors |
| **Total Extracted Road Area** | `16,366.52` | m² |
| **ROI Road Area Coverage** | `39.02%` | Observable transportation surface |
| **Total Extracted Road Length** | `370.74` | m |
| **Mean Estimated Corridor Width** | `16.43` | m |
| **Median Estimated Corridor Width** | `3.75` | m |
| **Geometric Validity** | `100.0%` | 100% compliant OGC/GEOS geometry |
| **Building Footprint Overlap Count** | `0` | Strict 0 building encroachment |
| **Building Overlap Area** | `0.0` | m² |
| **Extraction Latency** | `3.233s` | Classical CV pipeline execution time |

---

## 3. Road Extraction Architecture

The classical computer-vision baseline incorporates:
1. **Vegetation Suppression:** Excess Green Index ($ExG = 2G - R - B$) suppresses dense vegetative canopies and yard greenery.
2. **Spectral HSV Analysis:** Asphalt, packed earth, and unpaved sand roads exhibit moderate brightness ($V \in [65, 245]$) and low saturation ($S \le 70$).
3. **Texture Uniformity:** Sliding-window local variance filter separates smooth travel paths from granular vegetation and roof textures.
4. **Building Footprint Exclusion:** Structural footprint mask dilated by 1.0 m ($5\text{ px}$) strictly enforces 0 structural encroachment.
5. **Morphological Cleanup & Connected Components:** Removes isolated noise artifacts and bridges narrow road corridor segments while preserving elongated continuity ($L/W \ge 1.8$).

---

## 4. Visual Deliverables

The following 4 presentation-ready figures were produced at 300 DPI:
1. `01_input_image.png`: High-resolution aerial orthoimage.
2. `02_road_mask.png`: Soft likelihood heatmap $P(\text{road} \mid x)$ and binary access corridor mask.
3. `03_road_vector_overlay.png`: High-contrast vector polygons of extracted road network.
4. `04_buildings_roads_overlay.png`: Integrated AI Building Footprints + Extracted Road Corridors.

---

## 5. Artifacts and Exports

- Vector Polygons: `results/experiments/road_extraction/roads.geojson`
- Centerlines: `results/experiments/road_extraction/road_centerlines.geojson`
- Metrics: `results/experiments/road_extraction/road_metrics.csv`
- Topology Report: `results/experiments/road_extraction/road_topology_report.json`
