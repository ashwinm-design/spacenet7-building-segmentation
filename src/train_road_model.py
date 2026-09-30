"""
src/train_road_model.py

Executable script to prepare benchmark patches and train the Phase 13C U-Net ResNet18 road model.
Saves model checkpoint and comprehensive training metrics report.
"""

import os
import sys
import json
import time
from pathlib import Path
import pandas as pd

from road_dl import prepare_benchmark_patches
from road_dl.train import train_road_model


def run_training_experiment(
    raw_sample_dir: Path = Path("data/road_benchmark/deepglobe_sample"),
    patch_dir: Path = Path("data/road_benchmark/patches"),
    results_dir: Path = Path("results/experiments/road_dl"),
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
):
    print("=" * 70)
    print("SIH26012 PHASE 13C — DEEP-LEARNING ROAD MODEL TRAINING")
    print("=" * 70)

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Patch Slicing and Split Generation
    print("\n[1/3] Preparing and slicing benchmark patches (256x256)...")
    split_info = prepare_benchmark_patches(
        raw_dir=raw_sample_dir,
        output_dir=patch_dir,
        patch_size=256,
        train_ratio=0.80,
        seed=42,
    )
    print(f"      Total patches: {split_info['total_patches']} | Train: {split_info['train_count']} | Val: {split_info['val_count']}")

    # 2. Supervised Model Training
    print("\n[2/3] Training U-Net ResNet18 road segmentation model...")
    model, summary = train_road_model(
        patch_dir=patch_dir,
        split_info=split_info,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device="cpu",
        seed=42,
    )

    # 3. Export Training Metrics & Report
    print("\n[3/3] Exporting training metrics and report...")
    history_df = pd.DataFrame(summary["history"])
    history_df.to_csv(results_dir / "training_metrics.csv", index=False)

    best_m = summary["best_val_metrics"]

    report_md = f"""# SIH26012 Phase 13C — Deep-Learning Road Model Training Report

**Model Architecture:** U-Net with ResNet18 Encoder  
**Loss Function:** Compound BCE + Soft Dice Loss ($0.5 \\cdot \\text{{BCE}} + 0.5 \\cdot \\text{{Dice}}$)  
**Optimizer:** AdamW (Initial LR: {lr}, Weight Decay: 1e-4) with Cosine Annealing  
**Training Domain:** DeepGlobe Curated Benchmark ({split_info['total_patches']} patches of 256×256)  
**Training Hardware:** CPU ({summary['total_training_time_seconds']}s execution)  
**Best Epoch:** {summary['best_epoch']} / {epochs}  

---

## 1. Validation Performance on Benchmark

| Metric | Measured Value | Description |
|---|---|---|
| **Validation IoU** | **{best_m['iou']:.4f}** | Intersection over Union on held-out road patches |
| **Validation Dice (F1)** | **{best_m['dice']:.4f}** | Soft Dice coefficient |
| **Validation Precision**| **{best_m['precision']:.4f}** | True road pixels / predicted road pixels |
| **Validation Recall** | **{best_m['recall']:.4f}** | True road pixels / ground-truth road pixels |

---

## 2. Checkpoint Persistence

- **Saved Checkpoint:** `{summary['checkpoint_path']}`
- **Metrics Table:** `results/experiments/road_dl/training_metrics.csv`
"""

    with open(results_dir / "training_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Training completed! Best Val Dice: {summary['best_val_dice']:.4f}, IoU: {summary['best_val_iou']:.4f}")
    return model, summary


if __name__ == "__main__":
    run_training_experiment()
