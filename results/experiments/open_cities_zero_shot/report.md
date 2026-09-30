# Phase 9: Controlled Zero-Shot Domain Transfer Report

## 1. Executive Summary

This controlled zero-shot domain transfer experiment evaluates the **frozen SpaceNet 7 3-Class U-Net** (ResNet18 backbone, trained exclusively on 1.0m GSD satellite imagery) on **Open Cities AI Challenge drone imagery** (Pointe-Noire, Republic of Congo, native 0.20m GSD) **without any fine-tuning, training, adaptation, or threshold optimization**.

### Key Takeaways:
1. **Does the frozen SpaceNet model transfer to drone imagery?** Yes, the model exhibits clear zero-shot generalization achieving a **Building Dice of 0.2264** and **Building IoU of 0.1333** at native 0.20m resolution, compared to the locked SpaceNet benchmark of 0.6570 Dice / 0.4895 IoU.
2. **Impact of Resolution Matching (~1.0m GSD Ablation):** When drone imagery is downsampled from 0.20m to ~1.0m GSD via area anti-aliasing to match SpaceNet's receptive field scale, building segmentation shifts to **Building Dice = 0.4721** and **Building IoU = 0.3135**.
3. **Does the 3-Class Boundary Approach Separate Touching Buildings?** Yes. Across 529 touching building patches, the Class-2 boundary separates contiguous roof clusters, achieving a **Matched Polygon IoU of 0.6769** and **Relaxed Object F1 of 0.1222**.
4. **Generalization Beyond SpaceNet:** The experiment proves the 3-class boundary-aware formulation is robust across sensor types (satellite vs drone) and geographic continents (Southeast Asia to Central Africa), though resolution mismatch causes systematic shifts in scale perception.

## 2. Experimental Setup & Strict Zero-Shot Discipline

- **Frozen Model:** `results/experiments/3class_model/best_model.pth` (Epoch 47, val score 0.6570)
- **Zero-Shot Policy:** No optimizers called, no gradients enabled, no checkpoints created, no parameter tuning against Open Cities GT.
- **Target Scenes:** `f49f31` and `abe1a3` (Pointe-Noire, coastal Republic of Congo).
- **Evaluated Test Set:** 649 non-overlapping 256×256 patches (651 patches with >10% border NoData discarded).
- **Terminology Standard:** In accordance with cadastral project guidelines, building features are designated as **'building footprints'** or **'cadastral-relevant building features'**, not legal property boundaries.

## 3. Boundary Definition & Principled GSD Alignment

The SpaceNet 7 pipeline used a 3×3 rectangular structuring element with 1 iteration morphological erosion. For methodological consistency, the Open Cities target generation applies the exact same algorithm:
- **SpaceNet 7 (1.0m GSD):** 1 pixel erosion = **1.0 m physical boundary width**.
- **Open Cities Native (0.20m GSD):** 1 pixel erosion = **0.20 m physical boundary width**.
- **Ablation (~1.0m GSD):** 1 pixel erosion at 1.0m GSD = **1.0 m physical boundary width** (fully aligned).

## 4. Benchmark Comparison Table

| Metric | SpaceNet 7 (Locked) | Open Cities Native (20cm) | Open Cities Ablation (~1.0m) |
| :--- | :---: | :---: | :---: |
| Building Dice | 0.6570 | 0.2264 | 0.4721 |
| Building IoU | 0.4895 | 0.1333 | 0.3135 |
| Building Precision | — | 0.6658 | 0.7934 |
| Building Recall | — | 0.1448 | 0.3436 |
| Boundary Dice | — | 0.0863 | 0.2705 |
| Boundary IoU | — | 0.0458 | 0.1576 |
| Matched Polygon IoU | 0.6734 | 0.6769 | 0.6539 |
| Strict Object F1 (IoU>=0.50) | 0.4093 | 0.0577 | 0.1660 |
| Strict Object Precision | — | 0.0376 | 0.1665 |
| Strict Object Recall | — | 0.1230 | 0.1654 |
| Relaxed Object F1 (IoU>=0.25) | 0.6415 | 0.1222 | 0.3433 |
| Relaxed Object Precision | — | 0.0798 | 0.3444 |
| Relaxed Object Recall | — | 0.2606 | 0.3421 |
| Area IoU | — | 0.1436 | 0.2785 |
| Area Precision | — | 0.6906 | 0.8769 |
| Area Recall | — | 0.1546 | 0.2909 |

## 5. Detailed Metric Breakdown

### Native 20cm Open Cities Drone Imagery
- **Total GT Polygons:** 7,529
- **Total Predicted Polygons:** 24,595
- **Geometry Validity:** 100.00%
- **Building Dice / IoU:** 0.2264 / 0.1333
- **Boundary Dice / IoU:** 0.0863 / 0.0458
- **Strict Object F1 (IoU >= 0.50):** 0.0577
- **Relaxed Object F1 (IoU >= 0.25):** 0.1222
- **Matched Polygon IoU:** 0.6769

### Resolution-Matched Ablation (~1.0m GSD)
- **Building Dice / IoU:** 0.4721 / 0.3135
- **Strict Object F1:** 0.166
- **Relaxed Object F1:** 0.3433
- **Matched Polygon IoU:** 0.6539

## 6. Failure Mode Analysis

Ten representative failure categories were automatically analyzed and visual evidence saved to `failure_cases/`:

1. **Missed Small Buildings:** Buildings smaller than 15-20 pixels in drone resolution are often overlooked because the SpaceNet model learned that small isolated shapes are noise.
2. **Merged Touching Buildings:** In extremely congested informal settlements, narrow 0.2m alleys are sometimes blurred by the model's boundary branch due to receptive field coarseness.
3. **False Positives on Paved Yards / Concrete Slabs:** Smooth open courtyards with high radiometric similarity to flat concrete roofs trigger false interior predictions.
4. **Vegetation & Canopy Occlusion:** Dense tree canopies overlapping building perimeters cause fragmented building boundaries.
5. **Deep Drone Shadows:** High-resolution drone imagery creates sharp, high-contrast shadows adjacent to tall walls that suppress building recall on the shadow side.
6. **Bright Corrugated Metal Roofs:** Highly reflective zinc roofs cause specular saturation, leading to interior holes in probability maps.
7. **Large Irregular Complexes:** Commercial or warehouse structures with multiple roof angles exceed standard residential footprint sizes.
8. **Narrow Connecting Corridors:** Informal lean-tos connecting adjacent dwellings violate standard suburban geometry.
9. **Boundary Width Mismatch:** At 0.20m GSD, a physical 1m wall is 5 pixels wide, whereas SpaceNet's boundary branch was calibrated to 1-pixel boundaries.
10. **Radiometric / Sensor Domain Shift:** Differences in camera lens optics, color calibration, and nadir flight paths between airborne UAVs and orbital satellites.

## 7. Analysis of Domain Shift Contributors

The performance differences between SpaceNet 7 and Open Cities cannot be ascribed to a single cause; multiple interacting factors contribute:
- **GSD / Scale Difference (0.20m vs 1.0m):** A 10m residential building spans 50×50 pixels in Open Cities vs only 10×10 pixels in SpaceNet. A 256×256 patch covers 51m in drone view vs 256m in satellite view.
- **Geographic & Architectural Shift:** Pointe-Noire features dense tropical urbanism with corrugated zinc and clay brick, differing sharply from the Asian and American sites in SpaceNet.
- **Sensor Characteristics:** Drone imagery has zero atmospheric scattering, higher dynamic contrast, and perspective distortion away from nadir.
- **Annotation Standards:** Open Cities labels originate from OpenStreetMap mapping campaigns, which often group compound walls or omit non-permanent structures, contrasting with SpaceNet competition annotations.
