# Phase 11: Spatially Separated Dataset Split Report

## 1. Spatial Split Rationale & Anti-Leakage Protocol

To prevent visual and spatial data leakage between neighboring 256×256 evaluation patches, the Open Cities evaluation patches are partitioned into **spatially contiguous, non-overlapping sectors** for each scene. The **Validation Sector** is placed geographically between the **Training Sector** and **Test Sector**, acting as a 1,024-pixel physical buffer (>200 meters separation). This guarantees that no adjacent patches from the same neighborhood appear in both training and test.

## 2. Sector Boundaries & Geographic Partitioning

| Scene | Sector | Patch Column Range (`left`) | Physical Width | Patch Count | Share |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `f49f31` | **Train** (Western) | `left < 3328` | ~614 meters | 106 | 59.9% |
| `f49f31` | **Validation** (Buffer) | `3328 <= left < 4352` | ~205 meters | 35 | 19.8% |
| `f49f31` | **Test** (Eastern) | `left >= 4352` | ~358 meters | 36 | 20.3% |
| `abe1a3` | **Train** (Western) | `left < 4352` | ~819 meters | 277 | 58.7% |
| `abe1a3` | **Validation** (Buffer) | `4352 <= left < 5376` | ~205 meters | 95 | 20.1% |
| `abe1a3` | **Test** (Eastern) | `left >= 5376` | ~410 meters | 100 | 21.2% |
| **Combined** | **TRAIN** | — | — | **383** | **59.0%** |
| **Combined** | **VALIDATION** | — | — | **130** | **20.0%** |
| **Combined** | **TEST** | — | — | **136** | **21.0%** |

## 3. Dataset Size & Class Distribution Audit

| Metric | Training Split (59%) | Validation Split (20%) | Test Split (21%) |
| :--- | :---: | :---: | :---: |
| **Valid Patches** | 383 | 130 | 136 |
| **Empty Patches (0 bldgs)** | 22 (5.74%) | 7 (5.38%) | 8 (5.88%) |
| **Sparse Patches (1–5 bldgs)** | 21 (5.48%) | 6 (4.62%) | 16 (11.76%) |
| **Dense Patches (15+ bldgs)** | 278 (72.58%) | 98 (75.38%) | 71 (52.21%) |
| **Touching Building Patches** | 321 (83.81%) | 115 (88.46%) | 93 (68.38%) |
| **Mean Buildings / Patch** | 18.2 | 18.84 | 14.62 |
| **Mean Building Coverage** | 34.63% | 34.87% | 29.76% |
| **Class 0 (Background) Px** | 65.37% | 65.13% | 70.24% |
| **Class 1 (Interior) Px** | 30.44% | 30.4% | 26.18% |
| **Class 2 (Boundary) Px** | 4.2% | 4.47% | 3.57% |

## 4. Strict Protocol Confirmation

- **No Overlapping Bounding Boxes:** Spatial blocks are disjoint rectangular corridors.
- **Held-Out Test Set Discipline:** The 136 test patches are strictly held out and will be evaluated only once after all training runs terminate.
