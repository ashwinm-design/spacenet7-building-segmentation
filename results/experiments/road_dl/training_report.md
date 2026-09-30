# SIH26012 Phase 13C — Deep-Learning Road Model Training Report

**Model Architecture:** U-Net with ResNet18 Encoder  
**Loss Function:** Compound BCE + Soft Dice Loss ($0.5 \cdot \text{BCE} + 0.5 \cdot \text{Dice}$)  
**Optimizer:** AdamW (Initial LR: 0.001, Weight Decay: 1e-4) with Cosine Annealing  
**Training Domain:** DeepGlobe Curated Benchmark (96 patches of 256×256)  
**Training Hardware:** CPU (108.02s execution)  
**Best Epoch:** 10 / 10  

---

## 1. Validation Performance on Benchmark

| Metric | Measured Value | Description |
|---|---|---|
| **Validation IoU** | **0.3832** | Intersection over Union on held-out road patches |
| **Validation Dice (F1)** | **0.5541** | Soft Dice coefficient |
| **Validation Precision**| **0.5225** | True road pixels / predicted road pixels |
| **Validation Recall** | **0.5897** | True road pixels / ground-truth road pixels |

---

## 2. Checkpoint Persistence

- **Saved Checkpoint:** `results\experiments\road_dl\checkpoints\best_road_model.pth`
- **Metrics Table:** `results/experiments/road_dl/training_metrics.csv`
