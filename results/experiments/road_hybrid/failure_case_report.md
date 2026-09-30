# Phase 13D — Failure, Disagreement & Agreement Case Analysis

## Overview
Scientific integrity requires evaluating where models fail, where they disagree, and where hybrid fusion successfully remedies domain bottlenecks.
Below are 9 representative spatial cases identified automatically across the Open Cities Pointe-Noire ROI (`f49f31.tif`, $1024 \times 1024$ px).

---

### Case 1: Classical-Only Road Corridor
- **Classification**: `Disagreement / Method Specialization`
- **Pixel Window (Row, Col)**: `[217:337, 863:983]`
- **Diagnostic Description**: Broad paved/compacted access road captured reliably by classical HSV and texture filters, but missed by the deep learning model due to satellite training distribution gap.
- **Scientific & Planning Implication**: Demonstrates why classical CV remains indispensable as a robust topological trunk.
- **Detailed 4-Panel Crop**: [View Crop Image](crops/case_01.png)

---

### Case 2: Deep-Learning Only Road Segment
- **Classification**: `Disagreement / Method Specialization`
- **Pixel Window (Row, Col)**: `[97:217, 476:596]`
- **Diagnostic Description**: Narrow access corridor detected by convolutional contextual features that had low texture contrast or non-standard earth coloration.
- **Scientific & Planning Implication**: Shows deep learning recovering pathways beyond rigid spectral thresholds.
- **Detailed 4-Panel Crop**: [View Crop Image](crops/case_02.png)

---

### Case 3: Mutual Multi-Method Agreement
- **Classification**: `High-Confidence Consensus`
- **Pixel Window (Row, Col)**: `[572:692, 855:975]`
- **Diagnostic Description**: Unambiguous major thoroughfare with high spectral contrast, linear continuity, and clear building setbacks where both models exhibit 100% mutual consensus.
- **Scientific & Planning Implication**: Provides the highest-confidence foundation for property parcel boundaries.
- **Detailed 4-Panel Crop**: [View Crop Image](crops/case_03.png)

---

### Case 4: Deep-Learning False Positive (Rooftop Intrusion)
- **Classification**: `Failure Mode`
- **Pixel Window (Row, Col)**: `[725:845, 740:860]`
- **Diagnostic Description**: Deep learning model falsely predicting road corridor over corrugated metal rooftop due to flat specular reflection similarity to concrete roadway.
- **Scientific & Planning Implication**: Highlights why hard AI building exclusion is mandatory before parcel candidate generation.
- **Detailed 4-Panel Crop**: [View Crop Image](crops/case_04.png)

---

### Case 5: Classical False Negative (Shadowed Alleyway)
- **Classification**: `Failure Mode`
- **Pixel Window (Row, Col)**: `[98:218, 476:596]`
- **Diagnostic Description**: Narrow passageway between informal residences cast into deep building shadow; rejected by classical HSV brightness filter but recognized by DL context.
- **Scientific & Planning Implication**: Justifies hybrid fusion to prevent isolating interior dwellings from road access.
- **Detailed 4-Panel Crop**: [View Crop Image](crops/case_05.png)

---

### Case 6: Hybrid Topological Recovery (Bridged Gap)
- **Classification**: `Hybrid Benefit`
- **Pixel Window (Row, Col)**: `[96:216, 476:596]`
- **Diagnostic Description**: A fragmented gap in the classical corridor network successfully bridged and reinforced by hybrid probability fusion without encroaching on buildings.
- **Scientific & Planning Implication**: Reduces network fragmentation and directly improves parcel candidate road frontage.
- **Detailed 4-Panel Crop**: [View Crop Image](crops/case_06.png)

---

### Case 7: Tree-Canopy Occlusion Breakage
- **Classification**: `Failure Mode / Physical Barrier`
- **Pixel Window (Row, Col)**: `[328:448, 784:904]`
- **Diagnostic Description**: Dense tropical palm canopies overhanging unpaved street, obscuring the ground surface and causing both classical and neural models to fragment.
- **Scientific & Planning Implication**: Requires topological gap bridging heuristics to preserve route continuity across green canopies.
- **Detailed 4-Panel Crop**: [View Crop Image](crops/case_07.png)

---

### Case 8: Building-Adjacent Corridor (Setback Tension)
- **Classification**: `Spatial Boundary Tension`
- **Pixel Window (Row, Col)**: `[88:208, 241:361]`
- **Diagnostic Description**: High-density informal settlement scenario where road corridor passes within 0.5m of building walls, requiring exact 0.2m setback buffering.
- **Scientific & Planning Implication**: Prevents over-erosion of narrow alleys while strictly protecting building footprint integrity.
- **Detailed 4-Panel Crop**: [View Crop Image](crops/case_08.png)

---

### Case 9: Narrow Alleyway Recovery (< 2.5m Width)
- **Classification**: `Hybrid Benefit`
- **Pixel Window (Row, Col)**: `[96:216, 476:596]`
- **Diagnostic Description**: Sub-2.5m pedestrian pathway running between residential clusters, successfully recovered and polygonized as a secondary access branch.
- **Scientific & Planning Implication**: Enables the parcel engine to detect road-adjacency for previously landlocked structures.
- **Detailed 4-Panel Crop**: [View Crop Image](crops/case_09.png)

---
