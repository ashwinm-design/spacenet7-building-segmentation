# SIH26012 — System Limitations & Boundary Conditions

**Project:** SIH26012 AI-Assisted Urban Parcel Mapping  
**Date:** September 30, 2026  
**Document Classification:** Technical Transparency & Engineering Boundaries  

---

## 1. Cadastral Ground Truth & Legal Demarcation Limits

1. **Absence of Authoritative Parcel Ground Truth in Open Cities:**  
   The Open Cities AI dataset contains building footprint annotations but does **not** provide surveyed cadastral property boundaries or land ownership titles. Consequently, parcel candidate evaluation is based on geometric and topological plausibility (building containment, road exclusion, compactness) rather than legal survey accuracy.
2. **Preliminary Hypotheses Only:**  
   AI-generated parcel polygons are computational spatial hypotheses derived from road-constrained Voronoi partitioning. They **do not establish legal boundary rights, title ownership, or official registry entries**. Professional ground survey verification by a licensed cadastral surveyor remains legally mandatory.

---

## 2. Road Extraction Validation Limits

1. **No Surveyed Road Centerlines in Evaluated ROI:**  
   While Open Cities provides drone orthomosaics, authoritative vector road ground truth is absent for subregion `f49f31`. Road performance was evaluated via comparative topological analysis (Classical vs DL vs Phase 13D Hybrid Fusion) and connectivity metrics rather than ground-truth IoU.
2. **Dirt Roads & Low-Contrast Corridors:**  
   In arid or unpaved informal settlements, roads with identical spectral reflectance to surrounding bare soil can experience boundary fuzziness or false-negative gaps, which requires human verification.

---

## 3. Optical Imagery & Environmental Occlusions

1. **Vegetation & Tree-Canopy Occlusion:**  
   Dense tree canopies and overhanging foliage obscure building boundaries, small outbuildings, and pedestrian access corridors in optical imagery.
2. **Shadow Artifacts & Roof Reflectance:**  
   Tall structures cast shadows that shift apparent building contours, and highly reflective metallic roofs can saturate RGB sensor dynamic range, occasionally challenging edge detection.

---

## 4. Cross-Domain Generalization & Architectural Shift

1. **Domain Shift Across Regions:**  
   While the system demonstrated zero-shot generalization across distinct tiles (`f49f31` and `abe1a3`), models trained predominantly on low-to-medium density structures may experience reduced accuracy in dense high-rise urban cores, industrial complexes, or snow-covered winter scenes without targeted retraining.

---

## 5. Computational Geometry & Runtime Bottlenecks

1. **Parcel Generation Bottleneck:**  
   Empirical benchmarking reveals that Stage 7 (`PARCEL_GENERATION`) constitutes **58% to 60% of total pipeline runtime** (~20 to 27 seconds on CPU). Spatial difference operations subtracting buffered road networks from hundreds of Voronoi polygons are CPU-intensive.
2. **Browser-Based GIS Editing vs Cadastral CAD:**  
   The WebGIS interface provides interactive polygon vertex dragging, midpoint insertion, and candidate status updates suitable for verification and rapid corrections. However, it is not a full-featured civil surveying CAD tool and does not support precision coordinate geometry (COGO) traverse calculations or bearing-distance entry.

---

## 6. Summary of Engineering Integrity

These documented limitations represent the true state of autonomous spatial reconnaissance: AI accelerates survey preparation by up to 10x by identifying physical boundaries and access corridors, while human-in-the-loop oversight guarantees municipal legal compliance.
