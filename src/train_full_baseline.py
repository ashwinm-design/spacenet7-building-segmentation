"""
Phase 2: SpaceNet 7 Full Baseline Training Pipeline.

Architecture: U-Net with ResNet-18 ImageNet-pretrained encoder
Dataset: 256 Train, 64 Val, 64 Test (256x256 patches)
Loss: 50% BCEWithLogitsLoss + 50% DiceLoss
Optimizer: AdamW (lr=1e-3, weight_decay=1e-4) with ReduceLROnPlateau
Augmentation: Training only (flips, rotations, crop/scale, brightness/contrast, noise/blur)
Evaluation: Full held-out test evaluation, 12+ multi-category visual panels, failure analysis.
"""

import os
import sys
import time
import csv
import json
import argparse
import random
from typing import Dict, List, Tuple, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import BuildingDataset, get_dataloaders
from src.model import get_model, BCEDiceLoss


def compute_batch_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6):
    """Compute Dice, IoU, Precision, and Recall on batch tensors."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    tp = (preds_flat * targets_flat).sum().item()
    fp = (preds_flat * (1.0 - targets_flat)).sum().item()
    fn = ((1.0 - preds_flat) * targets_flat).sum().item()
    union = tp + fp + fn

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (union + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return dice, iou, precision, recall, tp, fp, fn


def plot_training_curves(metrics_csv: str, output_png: str):
    """Plot comprehensive train vs val training diagnostic curves."""
    epochs = []
    train_loss, val_loss = [], []
    train_dice, val_dice = [], []
    train_iou, val_iou = [], []
    train_prec, val_prec = [], []
    train_rec, val_rec = [], []

    with open(metrics_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_dice.append(float(row["train_dice"]))
            val_dice.append(float(row["val_dice"]))
            train_iou.append(float(row["train_iou"]))
            val_iou.append(float(row["val_iou"]))
            train_prec.append(float(row["train_precision"]))
            val_prec.append(float(row["val_precision"]))
            train_rec.append(float(row["train_recall"]))
            val_rec.append(float(row["val_recall"]))

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("#0b0f19")

    for ax in axes.flat:
        ax.set_facecolor("#141c2e")
        ax.tick_params(colors="#94a3b8")
        ax.spines["bottom"].set_color("#23304a")
        ax.spines["top"].set_color("#23304a")
        ax.spines["left"].set_color("#23304a")
        ax.spines["right"].set_color("#23304a")
        ax.grid(True, color="#23304a", linestyle="--", alpha=0.6)

    # 1. Loss
    axes[0, 0].plot(epochs, train_loss, color="#ef4444", linewidth=2.2, label="Train Loss")
    axes[0, 0].plot(epochs, val_loss, color="#f87171", linewidth=2.2, linestyle="--", label="Val Loss")
    axes[0, 0].set_title("BCE + Dice Loss", color="#fff", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch", color="#94a3b8")
    axes[0, 0].legend(facecolor="#141c2e", edgecolor="#23304a", labelcolor="#fff")

    # 2. Dice Score
    axes[0, 1].plot(epochs, train_dice, color="#10b981", linewidth=2.2, label="Train Dice")
    axes[0, 1].plot(epochs, val_dice, color="#34d399", linewidth=2.2, linestyle="--", label="Val Dice")
    axes[0, 1].set_title("Dice Coefficient", color="#fff", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch", color="#94a3b8")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].legend(facecolor="#141c2e", edgecolor="#23304a", labelcolor="#fff")

    # 3. IoU Score
    axes[1, 0].plot(epochs, train_iou, color="#3b82f6", linewidth=2.2, label="Train IoU")
    axes[1, 0].plot(epochs, val_iou, color="#60a5fa", linewidth=2.2, linestyle="--", label="Val IoU")
    axes[1, 0].set_title("Intersection over Union (Jaccard)", color="#fff", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch", color="#94a3b8")
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].legend(facecolor="#141c2e", edgecolor="#23304a", labelcolor="#fff")

    # 4. Precision & Recall (Val)
    axes[1, 1].plot(epochs, val_prec, color="#f59e0b", linewidth=2.2, label="Val Precision")
    axes[1, 1].plot(epochs, val_rec, color="#a855f7", linewidth=2.2, linestyle="--", label="Val Recall")
    axes[1, 1].set_title("Validation Precision & Recall", color="#fff", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch", color="#94a3b8")
    axes[1, 1].set_ylim(-0.02, 1.02)
    axes[1, 1].legend(facecolor="#141c2e", edgecolor="#23304a", labelcolor="#fff")

    plt.tight_layout()
    plt.savefig(output_png, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Run full evaluation on held-out test set and calculate per-sample metrics."""
    model.eval()
    per_sample_results = []
    total_loss = 0.0
    all_logits, all_targets = [], []

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    with torch.no_grad():
        for imgs, masks, names in test_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            loss = criterion(logits, masks)
            total_loss += loss.item() * imgs.size(0)

            all_logits.append(logits)
            all_targets.append(masks)

            # Per-sample metrics
            for i in range(imgs.size(0)):
                sub_logits = logits[i:i+1]
                sub_targets = masks[i:i+1]
                sub_dice, sub_iou, sub_prec, sub_rec, tp, fp, fn = compute_batch_metrics(sub_logits, sub_targets)

                gt_pixels = int(sub_targets.sum().item())
                total_pixels = sub_targets.size(-2) * sub_targets.size(-1)
                cov_pct = float(gt_pixels / total_pixels * 100)

                pred_prob = torch.sigmoid(sub_logits).squeeze().cpu().numpy()
                pred_binary = (pred_prob > 0.5).astype(np.uint8)
                pred_pixels = int(pred_binary.sum())

                per_sample_results.append({
                    "sample_name": names[i],
                    "coverage_pct": round(cov_pct, 2),
                    "gt_pixels": gt_pixels,
                    "pred_pixels": pred_pixels,
                    "dice": round(sub_dice, 4),
                    "iou": round(sub_iou, 4),
                    "precision": round(sub_prec, 4),
                    "recall": round(sub_rec, 4),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn)
                })

    cat_logits = torch.cat(all_logits, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)
    test_dice, test_iou, test_prec, test_rec, _, _, _ = compute_batch_metrics(cat_logits, cat_targets)
    avg_test_loss = total_loss / len(test_loader.dataset)

    test_metrics = {
        "test_loss": round(avg_test_loss, 5),
        "test_dice": round(test_dice, 4),
        "test_iou": round(test_iou, 4),
        "test_precision": round(test_prec, 4),
        "test_recall": round(test_rec, 4)
    }

    return test_metrics, per_sample_results


def select_diverse_visual_cases(per_sample_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Select at least 12 diverse test cases covering:
    - sparse area
    - dense urban area
    - small buildings
    - large buildings
    - touching/closely spaced
    - background-heavy/empty
    - moderate suburban
    """
    sorted_by_cov = sorted(per_sample_results, key=lambda x: x["coverage_pct"])
    sorted_by_iou = sorted(per_sample_results, key=lambda x: x["iou"])

    selected = []
    selected_names = set()

    def add_sample(sample, category):
        if sample["sample_name"] not in selected_names:
            s_copy = dict(sample)
            s_copy["case_category"] = category
            selected.append(s_copy)
            selected_names.add(sample["sample_name"])

    # 1. Empty / Zero building tiles
    empty_candidates = [s for s in sorted_by_cov if s["coverage_pct"] == 0.0]
    for s in empty_candidates[:2]:
        add_sample(s, "Empty / Background-Only Tile")

    # 2. Sparse Rural Homesteads (0.1% to 2.5% coverage)
    sparse_candidates = [s for s in sorted_by_cov if 0.1 <= s["coverage_pct"] <= 2.5]
    for s in sparse_candidates[:3]:
        add_sample(s, "Sparse Rural / Low Density")

    # 3. Dense Urban / Commercial (> 10% coverage)
    dense_candidates = [s for s in sorted_by_cov if s["coverage_pct"] >= 10.0]
    for s in dense_candidates[:3]:
        add_sample(s, "Dense Urban / High Density Clusters")

    # 4. Moderate Suburban Subdivision (3% to 8% coverage)
    mid_candidates = [s for s in sorted_by_cov if 3.0 <= s["coverage_pct"] <= 8.0]
    for s in mid_candidates[:3]:
        add_sample(s, "Suburban Residential / Closely Spaced Parcels")

    # 5. Best Prediction
    for s in reversed(sorted_by_iou):
        if len(selected) >= 12:
            break
        add_sample(s, "High-Accuracy Cadastral Delineation")

    # 6. Challenging / Low IoU sample
    for s in sorted_by_iou:
        if s["coverage_pct"] > 0:  # non-empty challenging
            add_sample(s, "Challenging Contrast / Partial Occlusion")
            break

    # Fill up to at least 12 if needed
    for s in sorted_by_cov:
        if len(selected) >= 14:
            break
        add_sample(s, "Representative Test Tile")

    return selected


def generate_visual_panels(
    model: nn.Module,
    selected_cases: List[Dict[str, Any]],
    splits_file: str,
    output_dir: str,
    device: torch.device
):
    """
    Generate the 6 required visualization components for selected test cases:
    1. Original aerial image
    2. Ground-truth mask
    3. Predicted probability map (magma heatmap)
    4. Thresholded predicted mask
    5. Cadastral Overlay
    6. Multi-panel side-by-side comparison
    """
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    model.eval()

    # Build path map from split file
    path_map = {}
    with open(splits_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                name = os.path.basename(parts[0])
                path_map[name] = (parts[0], parts[1])

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    with torch.no_grad():
        for case in selected_cases:
            name = case["sample_name"]
            if name not in path_map:
                continue

            img_path, mask_path = path_map[name]
            base_id = os.path.splitext(name)[0]

            raw_rgb = Image.open(img_path).convert("RGB")
            raw_mask = Image.open(mask_path).convert("L")

            # Tensor
            img_arr = np.array(raw_rgb, dtype=np.float32) / 255.0
            norm_img = (img_arr - mean) / std
            tensor_x = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

            logits = model(tensor_x)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_bin = (prob > 0.5).astype(np.uint8) * 255
            prob_uint8 = np.clip(prob * 255, 0, 255).astype(np.uint8)
            gt_arr = (np.array(raw_mask) > 127).astype(np.uint8) * 255

            # 1. Original
            raw_rgb.save(os.path.join(pred_dir, f"{base_id}_original.png"))
            # 2. Ground Truth Mask
            Image.fromarray(gt_arr).save(os.path.join(pred_dir, f"{base_id}_gt.png"))
            # 3. Probability Heatmap
            prob_color = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_MAGMA)
            prob_rgb = cv2.cvtColor(prob_color, cv2.COLOR_BGR2RGB)
            Image.fromarray(prob_rgb).save(os.path.join(pred_dir, f"{base_id}_prob.png"))
            # 4. Thresholded Mask
            Image.fromarray(pred_bin).save(os.path.join(pred_dir, f"{base_id}_pred.png"))

            # 5. Cadastral Overlay (RGB + predicted footprint translucent red + neon cyan contours)
            overlay = np.array(raw_rgb).copy()
            pred_fill = np.zeros_like(overlay)
            pred_fill[pred_bin > 0] = [255, 50, 50]  # Coral red fill
            overlay = cv2.addWeighted(overlay, 0.72, pred_fill, 0.28, 0)
            contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (40, 255, 120), 1, lineType=cv2.LINE_AA)  # Green contour
            Image.fromarray(overlay).save(os.path.join(pred_dir, f"{base_id}_overlay.png"))

            # 6. Side-by-Side Comprehensive Comparison
            # Layout: Original | Ground Truth | Prediction | Probability Map | Cadastral Overlay
            w, h = 256, 256
            header_h = 32
            comp_w = w * 5
            comp_h = h + header_h
            composite = np.zeros((comp_h, comp_w, 3), dtype=np.uint8)
            composite[:header_h, :] = [20, 24, 32]

            gt_3ch = cv2.cvtColor(gt_arr, cv2.COLOR_GRAY2RGB)
            pred_3ch = cv2.cvtColor(pred_bin, cv2.COLOR_GRAY2RGB)

            composite[header_h:, 0:w] = np.array(raw_rgb)
            composite[header_h:, w:w*2] = gt_3ch
            composite[header_h:, w*2:w*3] = pred_3ch
            composite[header_h:, w*3:w*4] = prob_rgb
            composite[header_h:, w*4:w*5] = overlay

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.40
            cv2.putText(composite, "1. Aerial Image", (10, 21), font, font_scale, (220, 225, 235), 1, cv2.LINE_AA)
            cv2.putText(composite, "2. Ground Truth", (w + 10, 21), font, font_scale, (220, 225, 235), 1, cv2.LINE_AA)
            cv2.putText(composite, "3. Prediction", (w * 2 + 10, 21), font, font_scale, (80, 240, 140), 1, cv2.LINE_AA)
            cv2.putText(composite, "4. Probability Map", (w * 3 + 10, 21), font, font_scale, (240, 180, 80), 1, cv2.LINE_AA)
            cv2.putText(composite, "5. Cadastral Overlay", (w * 4 + 10, 21), font, font_scale, (100, 220, 255), 1, cv2.LINE_AA)

            for i in range(1, 5):
                cv2.line(composite, (w * i, 0), (w * i, comp_h), (50, 58, 70), 1)

            Image.fromarray(composite).save(os.path.join(pred_dir, f"{base_id}_comparison.png"))


def run_failure_analysis(
    per_sample_results: List[Dict[str, Any]],
    output_dir: str
) -> Dict[str, Any]:
    """Perform automated failure analysis across test samples."""
    fa_dir = os.path.join(output_dir, "failure_analysis")
    os.makedirs(fa_dir, exist_ok=True)

    # 1. Save per-sample CSV
    csv_path = os.path.join(fa_dir, "test_evaluation_per_sample.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_name", "coverage_pct", "gt_pixels", "pred_pixels",
            "dice", "iou", "precision", "recall", "tp", "fp", "fn"
        ])
        writer.writeheader()
        writer.writerows(per_sample_results)

    # 2. Identify failure modes & outlier categories
    non_empty = [s for s in per_sample_results if s["coverage_pct"] > 0]
    empty = [s for s in per_sample_results if s["coverage_pct"] == 0]

    top_best = sorted(non_empty, key=lambda x: x["iou"], reverse=True)[:5]
    top_worst = sorted(non_empty, key=lambda x: x["iou"])[:5]
    high_fp = sorted(per_sample_results, key=lambda x: x["fp"], reverse=True)[:5]
    high_fn = sorted(non_empty, key=lambda x: x["fn"], reverse=True)[:5]

    # Empty tile TN analysis
    empty_false_positives = sum(s["pred_pixels"] for s in empty)
    empty_clean_count = sum(1 for s in empty if s["pred_pixels"] == 0)

    # 3. Density vs Performance Analysis
    # Group samples by coverage brackets
    brackets = {
        "empty (0%)": [s for s in per_sample_results if s["coverage_pct"] == 0],
        "sparse (0-3%)": [s for s in per_sample_results if 0 < s["coverage_pct"] <= 3.0],
        "moderate (3-8%)": [s for s in per_sample_results if 3.0 < s["coverage_pct"] <= 8.0],
        "dense (>8%)": [s for s in per_sample_results if s["coverage_pct"] > 8.0],
    }

    bracket_summary = {}
    for b_name, b_items in brackets.items():
        if b_items:
            bracket_summary[b_name] = {
                "count": len(b_items),
                "mean_dice": round(float(np.mean([s["dice"] for s in b_items])), 4),
                "mean_iou": round(float(np.mean([s["iou"] for s in b_items])), 4),
                "mean_precision": round(float(np.mean([s["precision"] for s in b_items])), 4),
                "mean_recall": round(float(np.mean([s["recall"] for s in b_items])), 4),
            }

    # 4. Generate Coverage vs IoU Diagnostic Scatter Plot
    plot_path = os.path.join(fa_dir, "coverage_vs_iou.png")
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#141c2e")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#23304a")
    ax.grid(True, color="#23304a", linestyle="--", alpha=0.6)

    covs = [s["coverage_pct"] for s in non_empty]
    ious = [s["iou"] for s in non_empty]
    dices = [s["dice"] for s in non_empty]

    scatter = ax.scatter(covs, ious, c=dices, cmap="viridis", s=70, edgecolors="#fff", alpha=0.85)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Dice Score", color="#94a3b8")
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8")

    ax.set_title("Test Building Density vs Segmentation IoU", color="#fff", fontsize=13, fontweight="bold")
    ax.set_xlabel("Building Coverage (% of Image)", color="#94a3b8", fontsize=11)
    ax.set_ylabel("Intersection over Union (IoU)", color="#94a3b8", fontsize=11)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close()

    report = {
        "total_test_samples": len(per_sample_results),
        "empty_samples_count": len(empty),
        "empty_samples_perfect_suppression": f"{empty_clean_count}/{len(empty)}",
        "top_best_predictions": top_best,
        "top_worst_predictions": top_worst,
        "high_false_positive_samples": high_fp,
        "high_false_negative_samples": high_fn,
        "density_bracket_performance": bracket_summary,
        "coverage_vs_iou_plot": plot_path,
        "per_sample_csv": csv_path
    }

    report_json_path = os.path.join(fa_dir, "failure_analysis_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def run_full_baseline_experiment(
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 1e-3,
    early_stopping_patience: int = 7,
    output_dir: str = "results/experiments/baseline_full"
):
    """Execute end-to-end Phase 2 Baseline Training & Evaluation."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 70)
    print("PHASE 2: SPACE NET 7 FULL BASELINE MODEL TRAINING & EVALUATION")
    print("=" * 70)
    print(f"Device:                {device}")
    print(f"Dataset Root:          data/processed_spacenet7")
    print(f"Max Epochs:            {epochs}")
    print(f"Early Stop Patience:   {early_stopping_patience} epochs")
    print(f"Batch Size:            {batch_size}")
    print(f"Initial LR:            {lr} (AdamW + ReduceLROnPlateau)")
    print(f"Augmentation:          ENABLED (Train only: Flips, 90 Rot, Crop, Photo)")
    print("-" * 70)

    # 1. Load DataLoaders
    # Note: BuildingDataset applies aerial augmentations when is_train=True
    loaders = get_dataloaders("data/processed_spacenet7", batch_size=batch_size, num_workers=0)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    print(f"DataLoaders Loaded:")
    print(f"  - Train: {len(train_loader.dataset)} patches ({len(train_loader)} batches)")
    print(f"  - Val:   {len(val_loader.dataset)} patches ({len(val_loader)} batches)")
    print(f"  - Test:  {len(test_loader.dataset)} patches ({len(test_loader)} batches)")
    print("-" * 70)

    # 2. Build Model
    model = get_model(architecture="unet", encoder_name="resnet18", in_channels=3, classes=1, pretrained=True)
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = "U-Net (ResNet18 ImageNet Encoder)"
    print(f"Model Architecture:    {model_name}")
    print(f"Trainable Parameters:  {param_count:,}")
    print("-" * 70)

    # 3. Loss & Optimizer
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # 4. Training Loop
    metrics_csv_path = os.path.join(output_dir, "metrics.csv")
    csv_file = open(metrics_csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow([
        "epoch", "train_loss", "train_dice", "train_iou", "train_precision", "train_recall",
        "val_loss", "val_dice", "val_iou", "val_precision", "val_recall", "lr", "epoch_time_s"
    ])

    best_val_dice = -1.0
    best_val_iou = -1.0
    best_epoch = 0
    best_model_path = os.path.join(output_dir, "best_model.pth")
    patience_counter = 0

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()

        # Training Phase
        model.train()
        train_loss = 0.0
        train_logits_list, train_targets_list = [], []

        for imgs, masks, _ in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_logits_list.append(logits.detach())
            train_targets_list.append(masks.detach())

        avg_train_loss = train_loss / len(train_loader.dataset)
        cat_tr_logits = torch.cat(train_logits_list, dim=0)
        cat_tr_targets = torch.cat(train_targets_list, dim=0)
        tr_dice, tr_iou, tr_prec, tr_rec, _, _, _ = compute_batch_metrics(cat_tr_logits, cat_tr_targets)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_logits_list, val_targets_list = [], []

        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = model(imgs)
                loss = criterion(logits, masks)

                val_loss += loss.item() * imgs.size(0)
                val_logits_list.append(logits)
                val_targets_list.append(masks)

        avg_val_loss = val_loss / len(val_loader.dataset)
        cat_val_logits = torch.cat(val_logits_list, dim=0)
        cat_val_targets = torch.cat(val_targets_list, dim=0)
        val_dice, val_iou, val_prec, val_rec, _, _, _ = compute_batch_metrics(cat_val_logits, cat_val_targets)

        epoch_time = time.time() - epoch_t0
        current_lr = optimizer.param_groups[0]["lr"]

        writer.writerow([
            epoch,
            f"{avg_train_loss:.5f}", f"{tr_dice:.4f}", f"{tr_iou:.4f}", f"{tr_prec:.4f}", f"{tr_rec:.4f}",
            f"{avg_val_loss:.5f}", f"{val_dice:.4f}", f"{val_iou:.4f}", f"{val_prec:.4f}", f"{val_rec:.4f}",
            f"{current_lr:.6f}", f"{epoch_time:.2f}"
        ])
        csv_file.flush()

        # Checkpoint Best Model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_iou = val_iou
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            is_best = " ★ BEST"
        else:
            patience_counter += 1
            is_best = ""

        scheduler.step(val_dice)

        print(f"Epoch [{epoch:2d}/{epochs:2d}] | "
              f"Train Loss: {avg_train_loss:.4f} Dice: {tr_dice:.4f} IoU: {tr_iou:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} Dice: {val_dice:.4f} IoU: {val_iou:.4f} "
              f"(Prec: {val_prec:.3f}, Rec: {val_rec:.3f}){is_best} [{epoch_time:.1f}s]")

        # Early Stopping Check
        if patience_counter >= early_stopping_patience:
            print(f"\n[Early Stopping] Validation Dice did not improve for {early_stopping_patience} consecutive epochs.")
            break

    csv_file.close()
    total_train_time = time.time() - t_start

    print("-" * 70)
    print(f"Training completed in {total_train_time:.1f}s ({total_train_time/60:.2f} mins)")
    print(f"Best Validation Epoch: {best_epoch} (Val Dice: {best_val_dice:.4f}, Val IoU: {best_val_iou:.4f})")

    # 5. Plot Learning Curves
    curve_png = os.path.join(output_dir, "training_curve.png")
    plot_training_curves(metrics_csv_path, curve_png)
    print(f"Learning curves saved to: {curve_png}")

    # 6. Evaluation on Held-Out Test Set (64 patches)
    print("-" * 70)
    print("Evaluating Best Model on Held-Out Test Set (64 patches)...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics, per_sample_results = evaluate_test_set(model, test_loader, criterion, device)

    print(f"Test Set Evaluation Results:")
    print(f"  - Test Loss:       {test_metrics['test_loss']:.5f}")
    print(f"  - Test Dice:       {test_metrics['test_dice']:.4f} ({test_metrics['test_dice']*100:.2f}%)")
    print(f"  - Test IoU:        {test_metrics['test_iou']:.4f} ({test_metrics['test_iou']*100:.2f}%)")
    print(f"  - Test Precision:  {test_metrics['test_precision']:.4f} ({test_metrics['test_precision']*100:.2f}%)")
    print(f"  - Test Recall:     {test_metrics['test_recall']:.4f} ({test_metrics['test_recall']*100:.2f}%)")

    # 7. Generate Visual Analysis Panels (>= 12 test samples)
    print("-" * 70)
    print("Generating Multi-Category Visual Panels for Test Cases...")
    diverse_cases = select_diverse_visual_cases(per_sample_results)
    generate_visual_panels(
        model, diverse_cases,
        splits_file="data/processed_spacenet7/splits/test.txt",
        output_dir=output_dir,
        device=device
    )
    print(f"Generated {len(diverse_cases)} multi-panel comparisons in {os.path.join(output_dir, 'predictions')}")

    # 8. Failure Analysis
    print("-" * 70)
    print("Running Automated Failure & Density Analysis...")
    fa_summary = run_failure_analysis(per_sample_results, output_dir)
    print(f"Failure analysis saved to: {os.path.join(output_dir, 'failure_analysis')}")

    # 9. Config JSON
    config_dict = {
        "experiment_name": "SpaceNet 7 Full Baseline",
        "model": model_name,
        "parameter_count": param_count,
        "input_size": [256, 256, 3],
        "epochs_trained": epoch,
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "initial_lr": lr,
        "weight_decay": 1e-4,
        "loss": "50% BCE + 50% Dice",
        "augmentations": ["horizontal_flip", "vertical_flip", "rotate_90", "random_crop_scale", "brightness_contrast", "gaussian_noise_blur"],
        "best_val_metrics": {
            "epoch": best_epoch,
            "val_dice": round(best_val_dice, 4),
            "val_iou": round(best_val_iou, 4)
        },
        "test_metrics": test_metrics,
        "training_duration_s": round(total_train_time, 2)
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    print("=" * 70)
    print("PHASE 2 BASELINE TRAINING COMPLETE!")
    print("=" * 70)

    return config_dict


def main():
    parser = argparse.ArgumentParser(description="SpaceNet 7 Full Baseline Training")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default 25)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default 16)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default 1e-3)")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--output-dir", default="results/experiments/baseline_full", help="Results directory")
    args = parser.parse_args()

    run_full_baseline_experiment(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stopping_patience=args.patience,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
