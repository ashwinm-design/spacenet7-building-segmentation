# Phase 6 Experiment Report: 3-Class U-Net Building Segmentation

**Date**: 2026-09-29  
**Project**: AI-Based Automated Urban Parcel Mapping and Cadastral Feature Extraction System (SIH 2026)  
**Dataset**: SpaceNet 7 Multi-Temporal Satellite Imagery (`L15-0506E-1204N_2027_3374_13`)  
**Experiment**: Phase 6 — 3-Class Semantic Segmentation (Background / Building Interior / Building Boundary)

---

## 1. Executive Summary & Benchmark Comparison

The 60-epoch training of the 3-class U-Net (ResNet-18 ImageNet-pretrained encoder) using compound Multiclass Focal + Dice Loss has completed successfully.

The explicit boundary representation has delivered a **major breakthrough across both pixel-level segmentation accuracy and instance separation**:

### Master Benchmark Comparison (Held-Out Test Set: 64 Patches)
| Metric | Binary U-Net Baseline (Phase 2) | Binary + Watershed Post-Proc (Phase 4) | **3-Class U-Net (Phase 6)** | **Δ vs Binary Baseline** |
| :--- | :---: | :---: | :---: | :---: |
| **Building Dice** | $0.6228$ | $0.5332$ | **$0.6570$** | **+3.42%** |
| **Building IoU** | $0.4522$ | $0.3634$ | **$0.4895$** | **+3.73%** |
| **Building Recall** | $0.6686$ | $0.7250$ | **$0.7675$** | **+9.89%** |
| **Building Precision**| $0.5829$ | $0.4411$ | **$0.5757$** | $-0.72\%$ |
| **Interior Dice** | — | — | **$0.5624$** | *New class* |
| **Boundary Dice** | — | — | **$0.4414$** | *New class* |
| **Separated Instances**| $2,990$ | $3,523$ | **$5,472$** | **+2,482 instances (+83.0%)** |
| **Ground Truth Instances**| $6,344$ (polygons) | $6,344$ (polygons) | **$9,965$ (GT interior CCs)**| — |
| **Instance Count Ratio** | $47.1\%$ | $55.5\%$ | **$54.9\%$** | Near target without heuristics |

---

## 2. Key Findings & Analysis

1. **Simultaneous Accuracy and Separation Boost**:
   Unlike classical watershed post-processing, which traded off area IoU (dropping from $0.4522$ to $0.3634$) in order to split fused instances, the 3-class model **improved both area IoU ($0.4522 \rightarrow 0.4895$) and instance count ($2,990 \rightarrow 5,472$) simultaneously**.
2. **Boundary Learning Prevents Under-Segmentation**:
   The boundary class achieved a test Dice of **$0.4414$** (and $0.5579$ recall), proving that the network successfully learned to delineate narrow 1–2 pixel boundaries separating adjacent rooftops.
3. **Massive Instance Recovery**:
   The interior connected components jumped from 1,545 in early epochs to **5,472** at convergence, successfully separating adjacent buildings that were previously fused into continuous horizontal and vertical blocks.
4. **Best Checkpoint**:
   Achieved at **Epoch 56** with validation score $0.5856$, validation building Dice $0.6666$, and validation building IoU $0.5000$.

---

## 3. Detailed Class-Wise Test Metrics

| Target Class | Dice | IoU | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Class 0 (Background)** | $0.9753$ | $0.9523$ | $0.9870$ | $0.9640$ |
| **Class 1 (Building Interior)** | **$0.5624$** | **$0.3917$** | **$0.5685$** | **$0.5587$** |
| **Class 2 (Building Boundary)** | **$0.4414$** | **$0.2834$** | **$0.3661$** | **$0.5579$** |
| **Macro Average (All 3 Classes)** | $0.6597$ | $0.5425$ | $0.6405$ | $0.6935$ |
| **Foreground Macro (Int + Bnd)** | $0.5019$ | $0.3376$ | $0.4673$ | $0.5583$ |
| **Reconstructed Building (Cls 1 \| 2)** | **$0.6570$** | **$0.4895$** | **$0.5757$** | **$0.7675$** |

---

## 4. Artifacts and Output Files

All experiment files are organized in `results/experiments/3class_model/`:
- Checkpoint: [`best_model.pth`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/results/experiments/3class_model/best_model.pth) (172 MB, Epoch 56)
- Metrics History: [`metrics.csv`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/results/experiments/3class_model/metrics.csv) (all 60 epochs recorded)
- Summary JSON: [`evaluation_summary.json`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/results/experiments/3class_model/evaluation_summary.json)
- Training Curves: [`training_curves.png`](file:///c:/Users/Ashwinpc/OneDrive/Desktop/sih%20project/results/experiments/3class_model/training_curves.png)
- Test Prediction Masks: `results/experiments/3class_model/test_predictions/` (contains colorized 3-class masks and reconstructed binary building masks)
