"""
SpaceNet 7 Phase 6: 3-Class Building Segmentation Training & Evaluation Pipeline.

Architecture:
  U-Net with ResNet-18 ImageNet-pretrained encoder configured for 3 classes:
    Class 0: Background
    Class 1: Building Interior
    Class 2: Building Boundary

Loss Function:
  Compound Multiclass Focal Loss + Multiclass Dice Loss:
    Class weights: [1.0 (bg), 4.0 (interior), 3.0 (boundary)]
    Gamma = 2.0
    Weighted without double-application.

Evaluation & Metrics:
  - Multiclass total loss, focal loss, dice loss
  - Class 0 (Background) Dice, IoU, Precision, Recall
  - Class 1 (Interior) Dice, IoU, Precision, Recall
  - Class 2 (Boundary) Dice, IoU, Precision, Recall
  - Macro Dice (mean of class Dices)
  - Reconstructed binary building mask: (pred == 1) | (pred == 2) -> Binary Dice, IoU, Precision, Recall
  - Instance separation analysis: Predicted Interior CCs vs Ground Truth Interior CCs
  - Strictly avoids misleading raw pixel accuracy.
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

# Ensure standard output uses UTF-8 to prevent Windows CP1252 charmap crashes
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.dataset import Building3ClassDataset, get_3class_dataloaders
from src.model import get_model, MulticlassFocalDiceLoss


# Color mapping for visual panels
COLOR_PALETTE = np.array([
    [30, 32, 40],      # Class 0: Background (Dark charcoal)
    [46, 204, 113],    # Class 1: Building Interior (Vibrant Emerald)
    [231, 76, 60],     # Class 2: Building Boundary (Vibrant Coral Red)
], dtype=np.uint8)


def colorize_3class(mask: np.ndarray) -> np.ndarray:
    """Map integer class mask {0, 1, 2} to RGB visualization."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        rgb[mask == c] = COLOR_PALETTE[c]
    return rgb


def compute_3class_batch_metrics(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> Dict[str, float]:
    """
    Compute comprehensive metrics on a batch of logits [B, 3, H, W] and targets [B, H, W]:
    - Per-class Dice, IoU, Precision, Recall for Class 0, 1, 2
    - Macro Dice
    - Reconstructed binary building metrics (Class 1 | Class 2)
    - Connected components instance counts
    """
    preds = torch.argmax(logits, dim=1) # [B, H, W]

    metrics = {}
    
    # ── 1. Per-Class Metrics ──
    class_names = ["bg", "interior", "boundary"]
    dices = []
    
    for c in range(3):
        p_c = (preds == c)
        t_c = (targets == c)

        tp = (p_c & t_c).sum().item()
        fp = (p_c & (~t_c)).sum().item()
        fn = ((~p_c) & t_c).sum().item()
        union = tp + fp + fn

        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        iou = (tp + eps) / (union + eps)
        prec = (tp + eps) / (tp + fp + eps)
        rec = (tp + eps) / (tp + fn + eps)

        metrics[f"{class_names[c]}_dice"] = float(dice)
        metrics[f"{class_names[c]}_iou"] = float(iou)
        metrics[f"{class_names[c]}_precision"] = float(prec)
        metrics[f"{class_names[c]}_recall"] = float(rec)
        dices.append(dice)

    metrics["macro_dice"] = float(np.mean(dices))
    metrics["fg_macro_dice"] = float(np.mean(dices[1:])) # Interior + Boundary

    # ── 2. Reconstructed Binary Building Metrics (Class 1 | Class 2) ──
    p_bldg = (preds == 1) | (preds == 2)
    t_bldg = (targets == 1) | (targets == 2)

    tp_b = (p_bldg & t_bldg).sum().item()
    fp_b = (p_bldg & (~t_bldg)).sum().item()
    fn_b = ((~p_bldg) & t_bldg).sum().item()
    union_b = tp_b + fp_b + fn_b

    metrics["bldg_dice"] = float((2.0 * tp_b + eps) / (2.0 * tp_b + fp_b + fn_b + eps))
    metrics["bldg_iou"] = float((tp_b + eps) / (union_b + eps))
    metrics["bldg_precision"] = float((tp_b + eps) / (tp_b + fp_b + eps))
    metrics["bldg_recall"] = float((tp_b + eps) / (tp_b + fn_b + eps))

    # ── 3. Instance Separation / Connected Components Analysis ──
    preds_np = preds.detach().cpu().numpy().astype(np.uint8)
    targets_np = targets.detach().cpu().numpy().astype(np.uint8)

    pred_cc_total = 0
    gt_cc_total = 0
    for b in range(preds_np.shape[0]):
        pred_int = (preds_np[b] == 1).astype(np.uint8)
        gt_int = (targets_np[b] == 1).astype(np.uint8)

        n_pred, _ = cv2.connectedComponents(pred_int)
        n_gt, _ = cv2.connectedComponents(gt_int)

        pred_cc_total += (n_pred - 1)
        gt_cc_total += (n_gt - 1)

    metrics["pred_interior_cc"] = float(pred_cc_total)
    metrics["gt_interior_cc"] = float(gt_cc_total)
    metrics["cc_ratio"] = float(pred_cc_total / (gt_cc_total + eps))

    return metrics


def evaluate_3class_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    """Evaluate model over an entire dataset split."""
    model.eval()
    total_loss = 0.0
    total_focal = 0.0
    total_dice = 0.0
    batch_count = 0

    accumulated = {
        "bg_dice": [], "bg_iou": [], "bg_precision": [], "bg_recall": [],
        "interior_dice": [], "interior_iou": [], "interior_precision": [], "interior_recall": [],
        "boundary_dice": [], "boundary_iou": [], "boundary_precision": [], "boundary_recall": [],
        "macro_dice": [], "fg_macro_dice": [],
        "bldg_dice": [], "bldg_iou": [], "bldg_precision": [], "bldg_recall": [],
        "pred_interior_cc": 0.0, "gt_interior_cc": 0.0
    }

    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            loss, focal, dice = criterion(logits, masks)

            total_loss += loss.item()
            total_focal += focal.item()
            total_dice += dice.item()
            batch_count += 1

            b_metrics = compute_3class_batch_metrics(logits, masks)
            for k in [
                "bg_dice", "bg_iou", "bg_precision", "bg_recall",
                "interior_dice", "interior_iou", "interior_precision", "interior_recall",
                "boundary_dice", "boundary_iou", "boundary_precision", "boundary_recall",
                "macro_dice", "fg_macro_dice",
                "bldg_dice", "bldg_iou", "bldg_precision", "bldg_recall"
            ]:
                accumulated[k].append(b_metrics[k])

            accumulated["pred_interior_cc"] += b_metrics["pred_interior_cc"]
            accumulated["gt_interior_cc"] += b_metrics["gt_interior_cc"]

    results = {
        "loss": total_loss / max(1, batch_count),
        "focal_loss": total_focal / max(1, batch_count),
        "dice_loss": total_dice / max(1, batch_count),
    }
    for k, v in accumulated.items():
        if isinstance(v, list):
            results[k] = float(np.mean(v))
        else:
            results[k] = float(v)

    results["cc_ratio"] = float(results["pred_interior_cc"] / (results["gt_interior_cc"] + 1e-6))
    return results


def run_sanity_test(
    dataset_dir: str = "data/processed_spacenet7_3class",
    encoder_name: str = "resnet18",
    device_str: str = "cpu"
) -> Dict[str, Any]:
    """
    Run 1-epoch training and validation sanity test on a small subset (16 samples).
    Validates all 6 critical sanity requirements:
    1. Output shape [B, 3, 256, 256]
    2. Labels contain only {0, 1, 2}
    3. Loss is finite
    4. Gradients are non-zero
    5. Predictions contain more than one class
    6. No NaN/Inf in model or loss
    """
    print("\n" + "=" * 60)
    print("PHASE 6: 1-EPOCH 3-CLASS MODEL TRAINING SANITY TEST")
    print("=" * 60)

    device = torch.device(device_str)

    # 1. Dataset loading
    splits_dir = os.path.join(dataset_dir, "splits")
    train_file = os.path.join(splits_dir, "train.txt")
    with open(train_file, "r", encoding="utf-8") as f:
        train_lines = [l.strip().split(",") for l in f if l.strip()]

    # Pick 16 diverse samples for the sanity subset
    subset_lines = train_lines[:16]
    img_paths = [l[0] for l in subset_lines]
    mask_paths = [l[1] for l in subset_lines]

    ds_sanity = Building3ClassDataset(img_paths, mask_paths, target_size=(256, 256), is_train=True)
    loader_sanity = DataLoader(ds_sanity, batch_size=4, shuffle=True)

    # 2. Build model
    print(f"Building U-Net with {encoder_name} encoder (3 output classes)...")
    model = get_model(
        architecture="unet",
        encoder_name=encoder_name,
        in_channels=3,
        classes=3,
        pretrained=True
    ).to(device)

    # 3. Loss & Optimizer
    criterion = MulticlassFocalDiceLoss(
        class_weights=(1.0, 4.0, 3.0),
        gamma=2.0,
        focal_weight=0.5,
        dice_weight=0.5
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 4. Check forward and backward on 1 mini-epoch
    model.train()
    checks = {
        "output_shape_correct": False,
        "labels_valid_subset": False,
        "loss_is_finite": False,
        "gradients_nonzero": False,
        "predictions_multiclass": False,
        "no_nan_or_inf": False
    }

    epoch_loss = 0.0
    all_pred_classes = set()

    for batch_idx, (imgs, masks, names) in enumerate(loader_sanity):
        imgs = imgs.to(device)
        masks = masks.to(device)

        # Check label set
        unique_labels = torch.unique(masks).tolist()
        if all(lbl in [0, 1, 2] for lbl in unique_labels):
            checks["labels_valid_subset"] = True

        optimizer.zero_grad()
        logits = model(imgs)

        # Check output shape
        if logits.shape == (imgs.shape[0], 3, 256, 256):
            checks["output_shape_correct"] = True

        loss, focal, dice = criterion(logits, masks)

        # Check finite loss
        if torch.isfinite(loss).item():
            checks["loss_is_finite"] = True

        loss.backward()

        # Check gradients
        total_grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        if total_grad > 0.0 and np.isfinite(total_grad):
            checks["gradients_nonzero"] = True

        optimizer.step()
        epoch_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_pred_classes.update(torch.unique(preds).cpu().tolist())

    if len(all_pred_classes) > 1:
        checks["predictions_multiclass"] = True

    # Check for NaN / Inf in parameters
    all_params_finite = all(torch.isfinite(p).all().item() for p in model.parameters())
    if all_params_finite:
        checks["no_nan_or_inf"] = True

    all_passed = all(checks.values())

    print("\nSanity Check Verification Results:")
    for k, v in checks.items():
        status = "PASS" if v else "FAIL"
        print(f"  - {k:<28}: [{status}] (value: {v})")

    print(f"  - Predicted classes in sanity pass: {sorted(list(all_pred_classes))}")
    print(f"  - Sanity epoch average loss: {epoch_loss / len(loader_sanity):.4f}")
    print(f"  - Overall Sanity Status: {'SUCCESS' if all_passed else 'FAILURE'}")
    print("=" * 60 + "\n")

    return {
        "status": "PASS" if all_passed else "FAIL",
        "checks": checks,
        "predicted_classes": sorted(list(all_pred_classes)),
        "sanity_loss": epoch_loss / len(loader_sanity)
    }


def plot_3class_training_curves(metrics_csv: str, output_png: str):
    """Plot multi-class training and validation curves."""
    epochs = []
    t_loss, v_loss = [], []
    t_macro_dice, v_macro_dice = [], []
    t_bldg_dice, v_bldg_dice = [], []
    v_int_dice, v_bnd_dice = [], []
    v_bldg_iou = []

    with open(metrics_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            t_loss.append(float(row["train_loss"]))
            v_loss.append(float(row["val_loss"]))
            t_macro_dice.append(float(row["train_macro_dice"]))
            v_macro_dice.append(float(row["val_macro_dice"]))
            t_bldg_dice.append(float(row["train_bldg_dice"]))
            v_bldg_dice.append(float(row["val_bldg_dice"]))
            v_int_dice.append(float(row["val_interior_dice"]))
            v_bnd_dice.append(float(row["val_boundary_dice"]))
            v_bldg_iou.append(float(row["val_bldg_iou"]))

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=200)
    fig.patch.set_facecolor("#0f1117")

    for ax in axes.flat:
        ax.set_facecolor("#1a1c24")
        ax.grid(True, linestyle="--", alpha=0.3, color="#444")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#444")

    # Plot 1: Total Loss
    axes[0, 0].plot(epochs, t_loss, label="Train Loss", color="#3498db", linewidth=2)
    axes[0, 0].plot(epochs, v_loss, label="Val Loss", color="#e74c3c", linewidth=2)
    axes[0, 0].set_title("1. Multiclass Focal+Dice Loss", color="white", fontweight="bold")
    axes[0, 0].legend(facecolor="#111", edgecolor="#333", labelcolor="white")

    # Plot 2: Macro Dice
    axes[0, 1].plot(epochs, t_macro_dice, label="Train Macro Dice", color="#2ecc71", linewidth=2)
    axes[0, 1].plot(epochs, v_macro_dice, label="Val Macro Dice", color="#f1c40f", linewidth=2)
    axes[0, 1].set_title("2. Macro Dice (All Classes)", color="white", fontweight="bold")
    axes[0, 1].legend(facecolor="#111", edgecolor="#333", labelcolor="white")

    # Plot 3: Interior vs Boundary Dice
    axes[1, 0].plot(epochs, v_int_dice, label="Val Interior Dice", color="#2ecc71", linewidth=2)
    axes[1, 0].plot(epochs, v_bnd_dice, label="Val Boundary Dice", color="#e74c3c", linewidth=2)
    axes[1, 0].set_title("3. Val Interior vs Boundary Separation", color="white", fontweight="bold")
    axes[1, 0].legend(facecolor="#111", edgecolor="#333", labelcolor="white")

    # Plot 4: Reconstructed Building Metrics
    axes[1, 1].plot(epochs, v_bldg_dice, label="Val Building Dice", color="#00ffff", linewidth=2)
    axes[1, 1].plot(epochs, v_bldg_iou, label="Val Building IoU", color="#9b59b6", linewidth=2)
    axes[1, 1].set_title("4. Reconstructed Building Footprint Overlap", color="white", fontweight="bold")
    axes[1, 1].legend(facecolor="#111", edgecolor="#333", labelcolor="white")

    plt.suptitle("PHASE 6: 3-CLASS U-NET TRAINING DYNAMICS", color="white", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def train_3class_model(
    dataset_dir: str = "data/processed_spacenet7_3class",
    output_dir: str = "results/experiments/3class_model",
    encoder_name: str = "resnet18",
    pretrained: bool = True,
    epochs: int = 60,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    gamma: float = 2.0,
    class_weights: Tuple[float, float, float] = (1.0, 4.0, 3.0),
    device_str: str = "cpu"
):
    """
    Run complete Phase 6 3-class U-Net training, validation, and test evaluation.
    """
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    val_pred_dir = os.path.join(output_dir, "val_predictions")
    test_pred_dir = os.path.join(output_dir, "test_predictions")
    for d in [val_pred_dir, test_pred_dir]:
        os.makedirs(d, exist_ok=True)

    device = torch.device(device_str)
    print("\n" + "=" * 70)
    print(f"STARTING PHASE 6: 3-CLASS U-NET TRAINING ({epochs} EPOCHS)")
    print(f"Device: {device} | Encoder: {encoder_name} | Pretrained: {pretrained}")
    print(f"Class Weights: Background={class_weights[0]}, Interior={class_weights[1]}, Boundary={class_weights[2]}")
    print("=" * 70)

    # 1. DataLoaders
    loaders = get_3class_dataloaders(dataset_dir, batch_size=batch_size, num_workers=0)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    print(f"Loaded datasets: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")

    # 2. Model
    model = get_model(
        architecture="unet",
        encoder_name=encoder_name,
        in_channels=3,
        classes=3,
        pretrained=pretrained
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.grad_fn is None or True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")

    # 3. Loss & Optimizer
    criterion = MulticlassFocalDiceLoss(
        class_weights=class_weights,
        gamma=gamma,
        focal_weight=0.5,
        dice_weight=0.5
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    # Metrics CSV tracking
    metrics_csv = os.path.join(output_dir, "metrics.csv")
    csv_fields = [
        "epoch", "train_loss", "train_focal", "train_dice_loss",
        "train_macro_dice", "train_fg_macro_dice", "train_bldg_dice", "train_bldg_iou",
        "val_loss", "val_focal", "val_dice_loss",
        "val_bg_dice", "val_interior_dice", "val_boundary_dice",
        "val_macro_dice", "val_fg_macro_dice",
        "val_bldg_dice", "val_bldg_iou", "val_bldg_prec", "val_bldg_rec",
        "val_pred_cc", "val_gt_cc", "val_cc_ratio", "lr"
    ]
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_fields)

    best_val_score = -1.0
    best_epoch = 0
    best_model_path = os.path.join(output_dir, "best_model.pth")

    # Training Loop
    for epoch in range(1, epochs + 1):
        t_ep_start = time.time()
        model.train()
        t_loss, t_focal, t_dice_loss = 0.0, 0.0, 0.0
        t_macro_dices, t_fg_macro_dices = [], []
        t_bldg_dices, t_bldg_ious = [], []
        n_batches = 0

        for imgs, masks, _ in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss, focal, dice = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            t_focal += focal.item()
            t_dice_loss += dice.item()
            n_batches += 1

            with torch.no_grad():
                b_metrics = compute_3class_batch_metrics(logits, masks)
                t_macro_dices.append(b_metrics["macro_dice"])
                t_fg_macro_dices.append(b_metrics["fg_macro_dice"])
                t_bldg_dices.append(b_metrics["bldg_dice"])
                t_bldg_ious.append(b_metrics["bldg_iou"])

        # Validation pass
        val_res = evaluate_3class_epoch(model, val_loader, criterion, device)
        cur_lr = optimizer.param_groups[0]["lr"]

        # Track score (reconstructed building Dice + foreground macro Dice)
        val_score = (val_res["bldg_dice"] + val_res["fg_macro_dice"]) / 2.0
        scheduler.step(val_score)

        is_best = val_score > best_val_score
        if is_best:
            best_val_score = val_score
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_score": best_val_score,
                "metrics": val_res
            }, best_model_path)

        train_loss_avg = t_loss / n_batches
        train_focal_avg = t_focal / n_batches
        train_dice_loss_avg = t_dice_loss / n_batches
        train_macro_dice_avg = float(np.mean(t_macro_dices))
        train_fg_macro_dice_avg = float(np.mean(t_fg_macro_dices))
        train_bldg_dice_avg = float(np.mean(t_bldg_dices))
        train_bldg_iou_avg = float(np.mean(t_bldg_ious))

        # Log row
        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, round(train_loss_avg, 4), round(train_focal_avg, 4), round(train_dice_loss_avg, 4),
                round(train_macro_dice_avg, 4), round(train_fg_macro_dice_avg, 4),
                round(train_bldg_dice_avg, 4), round(train_bldg_iou_avg, 4),
                round(val_res["loss"], 4), round(val_res["focal_loss"], 4), round(val_res["dice_loss"], 4),
                round(val_res["bg_dice"], 4), round(val_res["interior_dice"], 4), round(val_res["boundary_dice"], 4),
                round(val_res["macro_dice"], 4), round(val_res["fg_macro_dice"], 4),
                round(val_res["bldg_dice"], 4), round(val_res["bldg_iou"], 4),
                round(val_res["bldg_precision"], 4), round(val_res["bldg_recall"], 4),
                int(val_res["pred_interior_cc"]), int(val_res["gt_interior_cc"]),
                round(val_res["cc_ratio"], 4), f"{cur_lr:.1e}"
            ])

        ep_time = time.time() - t_ep_start
        best_marker = " [BEST]" if is_best else ""
        print(f"Epoch {epoch:02d}/{epochs:02d} [{ep_time:.1f}s] — "
              f"Train Loss: {train_loss_avg:.4f}, Bldg Dice: {train_bldg_dice_avg:.4f} | "
              f"Val Loss: {val_res['loss']:.4f}, Int Dice: {val_res['interior_dice']:.4f}, "
              f"Bnd Dice: {val_res['boundary_dice']:.4f}, Bldg Dice: {val_res['bldg_dice']:.4f}{best_marker}")

    # Generate training curve plots
    curves_png = os.path.join(output_dir, "training_curves.png")
    plot_3class_training_curves(metrics_csv, curves_png)

    # 4. Final Evaluation on Held-Out Test Set
    print("\n" + "=" * 70)
    print("FINAL HELD-OUT TEST EVALUATION (Using Best Checkpoint)")
    print("=" * 70)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_res = evaluate_3class_epoch(model, test_loader, criterion, device)

    # Save test prediction masks
    with torch.no_grad():
        for imgs, masks, names in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)

            for b in range(preds.shape[0]):
                pname = names[b].replace(".png", "")
                pred_color = colorize_3class(preds[b])
                Image.fromarray(pred_color).save(os.path.join(test_pred_dir, f"{pname}_pred_3class.png"))

                # Reconstructed binary building mask (0, 255)
                pred_bin = ((preds[b] == 1) | (preds[b] == 2)).astype(np.uint8) * 255
                Image.fromarray(pred_bin).save(os.path.join(test_pred_dir, f"{pname}_pred_binary.png"))

    # Config JSON
    config_dict = {
        "architecture": "unet",
        "encoder": encoder_name,
        "pretrained": pretrained,
        "classes": 3,
        "class_names": ["background", "interior", "boundary"],
        "class_weights": list(class_weights),
        "gamma": gamma,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "device": device_str,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "elapsed_seconds": round(time.time() - t_start, 2)
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # Evaluation Summary JSON
    summary_dict = {
        "best_epoch": best_epoch,
        "val_metrics_at_best": checkpoint["metrics"],
        "test_metrics": test_res
    }
    with open(os.path.join(output_dir, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    print("\nTraining and Evaluation Completed Successfully!")
    print(f"Best Checkpoint: {best_model_path} (Epoch {best_epoch})")
    print(f"Test Bldg Dice: {test_res['bldg_dice']:.4f} | Test Bldg IoU: {test_res['bldg_iou']:.4f}")
    print(f"Test Interior Dice: {test_res['interior_dice']:.4f} | Test Boundary Dice: {test_res['boundary_dice']:.4f}")
    print(f"Test Interior CCs: {int(test_res['pred_interior_cc'])} (GT: {int(test_res['gt_interior_cc'])})")
    print("=" * 70 + "\n")

    return summary_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit and Training for Phase 6 3-Class U-Net")
    parser.add_argument("--dataset_dir", type=str, default="data/processed_spacenet7_3class")
    parser.add_argument("--output_dir", type=str, default="results/experiments/3class_model")
    parser.add_argument("--encoder", type=str, default="resnet18")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--class_weights", nargs=3, type=float, default=[1.0, 4.0, 3.0])
    parser.add_argument("--loss", type=str, default="focal_dice")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sanity_test_only", action="store_true", help="Run 1-epoch sanity test only")
    args = parser.parse_args()

    if args.sanity_test_only:
        res = run_sanity_test(
            dataset_dir=args.dataset_dir,
            encoder_name=args.encoder,
            device_str=args.device
        )
        if res["status"] != "PASS":
            sys.exit(1)
        sys.exit(0)
    else:
        train_3class_model(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            encoder_name=args.encoder,
            pretrained=args.pretrained,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            gamma=args.gamma,
            class_weights=tuple(args.class_weights),
            device_str=args.device
        )

