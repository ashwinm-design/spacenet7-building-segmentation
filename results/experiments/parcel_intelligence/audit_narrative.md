# PHASE 14A — PARCEL INTELLIGENCE AUDIT NARRATIVE

## Population Summary

The Phase 14A audit evaluated **254 AI-generated preliminary parcel candidates**.
These candidates were produced by the SIH26012 hybrid pipeline (Phase 13D) using
boundary-aware U-Net building segmentation, hybrid classical+DL road extraction,
and Voronoi-based spatial partitioning.

### Confidence Distribution

| Statistic | Value |
|-----------|-------|
| Mean Confidence | 0.8814 |
| Median Confidence | 0.8946 |
| Std. Deviation | 0.0776 |
| Min Confidence | 0.6254 |
| Max Confidence | 1.0000 |
| P25 | 0.8295 |
| P75 | 0.9395 |

### Band Distribution

| Band | Count | Percentage |
|------|-------|------------|
| HIGH | 237 | 93.3% |
| MODERATE | 17 | 6.7% |
| LOW | 0 | 0.0% |
| VERY_LOW | 0 | 0.0% |

### Uncertainty Flags

- **82.3%** of candidates have at least one uncertainty flag.
- **327** total flags raised across all candidates.

| Flag Type | Count |
|-----------|-------|
| MULTI_SOURCE_CONFLICT | 169 |
| WEAK_BUILDING_EVIDENCE | 115 |
| UNDERSIZED_PARCEL | 30 |
| WEAK_ROAD_EVIDENCE | 11 |
| NEEDS_BOUNDARY_REVIEW | 2 |

### Top Performing Candidates

- Parcel 29: confidence=1.0000, strengths=[geometry_quality, building_evidence, road_access, topological_plausibility]
- Parcel 33: confidence=1.0000, strengths=[geometry_quality, building_evidence, road_access, topological_plausibility]
- Parcel 53: confidence=1.0000, strengths=[geometry_quality, building_evidence, road_access, topological_plausibility]
- Parcel 84: confidence=1.0000, strengths=[geometry_quality, building_evidence, road_access, topological_plausibility]
- Parcel 159: confidence=1.0000, strengths=[geometry_quality, building_evidence, road_access, topological_plausibility]

### Candidates Requiring Review

- Parcel 162: confidence=0.7180, flags=[WEAK_BUILDING_EVIDENCE]
- Parcel 182: confidence=0.7046, flags=[WEAK_BUILDING_EVIDENCE, WEAK_ROAD_EVIDENCE, MULTI_SOURCE_CONFLICT]
- Parcel 52: confidence=0.7020, flags=[WEAK_BUILDING_EVIDENCE, MULTI_SOURCE_CONFLICT]
- Parcel 85: confidence=0.6954, flags=[WEAK_BUILDING_EVIDENCE, MULTI_SOURCE_CONFLICT]
- Parcel 230: confidence=0.6863, flags=[WEAK_BUILDING_EVIDENCE, MULTI_SOURCE_CONFLICT]

---

**DISCLAIMER:** All confidence scores measure AI inference quality and geometric
plausibility. They do NOT represent legal cadastral certainty or property ownership.
All boundaries are preliminary AI-assisted candidates requiring professional survey validation.