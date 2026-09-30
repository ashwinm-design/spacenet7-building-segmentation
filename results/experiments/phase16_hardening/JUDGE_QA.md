# SIH26012 — Judge Talking Points & Technical Q&A

**Project:** SIH26012 AI-Assisted Urban Parcel Mapping  
**Guidance:** Every answer strictly distinguishes **MEASURED FACT**, **DESIGN DECISION**, **LIMITATION**, and **FUTURE WORK**.

---

### Q1. Why not just use Google Maps or OpenStreetMap?
- **DESIGN DECISION:** Commercial tile services like Google Maps do not permit raw pixel access for machine learning inference, prohibit commercial export of derived vector layers under standard terms, and provide low spatial resolution in developing or peri-urban areas.
- **MEASURED FACT:** Open Cities drone imagery provides 5–10 cm ground sampling distance (GSD), whereas standard web satellite tiles rarely exceed 30–50 cm GSD, obscuring individual building separation and narrow access paths.
- **LIMITATION:** OpenStreetMap lacks coverage for over 70% of informal settlements and rapidly growing peri-urban wards.
- **FUTURE WORK:** OSM vector layers can be integrated as optional prior constraints where available.

---

### Q2. Why use drone/aerial imagery instead of satellite?
- **MEASURED FACT:** Satellite imagery suffers from off-nadir tilt angles and atmospheric haze, leading to building footprint displacement. Very high-resolution drone orthomosaics provide true nadir orthorectification with sub-decimeter resolution.
- **DESIGN DECISION:** Drone flights can be commissioned on demand by municipal surveyors following property developments, offering rapid recency compared to satellite revisit schedules.
- **LIMITATION:** Drone surveys require airspace clearance, pilot deployment, and weather considerations.
- **FUTURE WORK:** Support multimodal input combining periodic satellite monitoring with targeted drone flights.

---

### Q3. How were buildings trained?
- **MEASURED FACT:** The model is a 3-Class Convolutional Neural Network (ResNet34 U-Net) predicting Background, Building Body, and Building Boundary.
- **DESIGN DECISION:** We adopted a 3-class target formulation rather than binary segmentation because binary masks cause adjacent, touching buildings to merge into large contiguous blobs during polygonization.
- **MEASURED FACT:** On SpaceNet 7, the 3-class approach maintained building boundary separation across 254 instances in dense clusters.
- **LIMITATION:** Severe building roof shadow or vegetative overhang can degrade boundary precision.

---

### Q4. What datasets were used?
- **MEASURED FACT:** SpaceNet 7 Multi-Temporal Urban Development Challenge dataset (for foundational 3-class segmentation training) and Open Cities AI Challenge raw orthomosaics (Point-Noire `f49f31` and `abe1a3` in UTM EPSG:32732 for zero-shot adaptation and demo validation).
- **DESIGN DECISION:** SpaceNet 7 provided high-quality verified polygonal annotations with temporal variations; Open Cities provided challenging African drone imagery with informal structures.

---

### Q5. Why SpaceNet?
- **MEASURED FACT:** SpaceNet 7 contains over 40,000 km² of imagery across multiple continents with planet-scale geographic diversity and professional GIS-audited building polygon footprints.
- **DESIGN DECISION:** SpaceNet allowed training a generalized feature extractor before testing on localized drone orthomosaics.

---

### Q6. Why Open Cities?
- **DESIGN DECISION:** To prove that our models generalize to real-world, resource-constrained municipal environments characterized by unpaved roads, metal roofs, and organic settlement layouts.
- **MEASURED FACT:** Open Cities tiles (`f49f31` and `abe1a3`) feature high-density informal dwellings where traditional cadastral maps are completely absent.

---

### Q7. How are roads extracted?
- **MEASURED FACT:** Roads are extracted via a dual-path pipeline:
  1. Classical CV: morphological opening, Otsu adaptive thresholding, and building mask exclusion.
  2. Deep Learning: U-Net with ResNet18 backbone trained on road benchmark patches.
- **DESIGN DECISION:** Combining classical and DL paths prevents total failure when one modality underperforms.

---

### Q8. Why combine classical CV and Deep Learning for roads?
- **DESIGN DECISION:** Classical CV reliably identifies physical, high-contrast open corridors and dirt lanes without hallucination, while DL identifies learned semantic road continuity.
- **MEASURED FACT:** In Phase 13D hybrid fusion, classical CV generated 6 primary corridors and DL recovered 41 auxiliary candidate segments, producing 1,571 meters of connected road network.
- **LIMITATION:** Dirt roads that blend into surrounding bare soil require manual threshold tuning.

---

### Q9. How are preliminary parcels generated?
- **DESIGN DECISION:** In the absence of cadastral ground truth, parcels are generated using computational spatial geometry: road-constrained Voronoi tessellation seeded by building centroids.
- **MEASURED FACT:** Every parcel polygon is clipped against the buffered road corridor network and repaired with Shapely `make_valid`, guaranteeing **100.0% geometry validity** and zero road overlap.
- **LIMITATION:** Voronoi boundaries are geometric spatial approximations, not legal property demarcations.

---

### Q10. Are these legal cadastral boundaries?
- **MEASURED FACT & LEGAL SCOPE:** **NO.** These are preliminary spatial candidates designed for planning and survey assistance.
- **DESIGN DECISION:** We prominently display a mandatory legal disclaimer across the WebGIS UI, API responses, and export deliverables: *"AI-generated parcel boundaries are preliminary spatial candidates and require professional cadastral/survey verification. Review status does not establish legal ownership or boundary rights."*
- **FUTURE WORK:** Integration with state Land Administration Domain Model (LADM / ISO 19152) standards.

---

### Q11. How is human verification handled?
- **DESIGN DECISION:** Through Phase 14B Human-in-the-Loop WebGIS.
- **MEASURED FACT:** Surveyors can inspect candidate metrics, accept high-confidence candidates, flag complex candidates as `NEEDS_REVIEW`, or interactively drag polygon vertices in edit mode.
- **MEASURED FACT:** Every action appends to an immutable chronological audit trail (`parcel_reviews_audit.json`) with timestamps, user notes, and before/after SHA-256 geometry hashes.

---

### Q12. What happens when the AI makes an error?
- **MEASURED FACT:** Low-confidence or geometrically irregular candidates are automatically assigned uncertainty flags (`WEAK_BUILDING_EVIDENCE`, `MULTI_SOURCE_CONFLICT`, `ZERO_FRONTAGE`, `HIGH_COVERAGE`).
- **DESIGN DECISION:** In the review queue, candidates default to `UNREVIEWED`. The reviewer can adjust vertices live or reject spurious candidates, preventing erroneous AI hypotheses from entering the municipal GIS.

---

### Q13. How does the system scale to large cities?
- **MEASURED FACT:** Inference operates via 256×256 sliding window tiles with 192-pixel stride and cosine window blending.
- **MEASURED FACT:** A 1024×1024 subregion (covering ~250 buildings and 1.5 km of roads) processes in ~38.8 seconds on standard CPU hardware.
- **FUTURE WORK:** Horizontal scaling via Celery/Redis worker pools and GPU acceleration (NVIDIA TensorRT) to process gigapixel orthomosaics in under 5 minutes.

---

### Q14. What happens when tested on a different city?
- **MEASURED FACT:** In Phase 15 generalization testing on tile `abe1a3` (a separate urban region in Point-Noire), the pipeline successfully extracted 246 buildings, 8 road corridors (2,063 m), and 246 parcel candidates with 100.0% geometry validity and 89.1% mean confidence without altering any hyperparameters or code.
- **LIMITATION:** Severe architectural domain shift (e.g., snow cover or high-rise skyscrapers) would require domain fine-tuning.

---

### Q15. What is the biggest current limitation?
- **MEASURED FACT & BOTTLENECK:** Stage 7 (`PARCEL_GENERATION`) accounts for ~58–60% of total pipeline runtime (20–27 seconds) due to complex computational polygon intersection and difference operations across 250+ geometries.
- **LIMITATION:** Lack of ground-truth land title boundaries in Open Cities prevents legal boundary accuracy evaluation.

---

### Q16. What would be needed for commercial / production deployment?
- **FUTURE WORK:**
  1. Integration with municipal GIS databases (PostGIS / GeoServer).
  2. Mobile field surveyor application for on-ground GPS boundary confirmation.
  3. Support for multi-user role-based access control (Surveyor, Senior Officer, Registrar).
  4. GPU cluster deployment with tile cache services.
