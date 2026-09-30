#!/usr/bin/env python3
"""
src/train_adaptation_model.py

Phase 11: Controlled Open Cities Domain Adaptation Training Pipeline
Trains the 3-Class U-Net (ResNet-18) on Open Cities drone imagery using
the spatially separated 60/20/20 split.

Experiments:
  1. Sanity Check (Section 10): 4-batch gradient & shape validation.
  2. Primary Domain Adaptation (Section 7, 8, 11):
     - Initialized from frozen SpaceNet 7 checkpoint (results/experiments/3class_model/best_model.pth)
     - AdamW optimizer (lr=1e-4, weight_decay=1e-4)
     - Multiclass Focal (gamma=2.0) + Dice Loss (weights=[1.0, 4.0, 3.0])
     - Max 40 epochs, early stopping patience=7 on validation Building Dice.
  3. Controlled Ablation (Section 16):
     - Initialized from ImageNet directly (NO SpaceNet pretraining)
     - Identical dataset splits, augmentation, optimizer, lr, loss, epochs.
"""

import os
import sys
import time
import csv
import json
import argparse
import random
import copy
import warnings
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import get_model, MulticlassFocalDiceLoss

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════
# Configuration & Constants
# ════════════════════════════════════════════════════════════════════
SPACENET_CKPT = "results/experiments/3class_model/best_model.pth"
OUTPUT_DIR = "results/experiments/open_cities_adaptation"
SPLITS_DIR = os.path.join(OUTPUT_DIR, "dataset", "splits")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

AUGMENTATION_CONFIG = {
    "horizontal_flip": True,
    "vertical_flip": True,
    "rotate_90": True,
    "random_crop_scale": (0.85, 1.0),
    "brightness_range": (0.85, 1.15),
    "contrast_range": (0.85, 1.15),
    "gaussian_noise_prob": 0.20,
    "gaussian_noise_std": 0.02,
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ════════════════════════════════════════════════════════════════════
# Dataset & Augmentation
# ════════════════════════════════════════════════════════════════════
def apply_augmentations(img_arr: np.ndarray, mask_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Moderate spatial and photometric augmentations preserving small structures."""
    # 1. Horizontal flip
    if AUGMENTATION_CONFIG["horizontal_flip"] and random.random() > 0.5:
        img_arr = np.fliplr(img_arr).copy()
        mask_arr = np.fliplr(mask_arr).copy()

    # 2. Vertical flip
    if AUGMENTATION_CONFIG["vertical_flip"] and random.random() > 0.5:
        img_arr = np.flipud(img_arr).copy()
        mask_arr = np.flipud(mask_arr).copy()

    # 3. 90-degree rotations
    if AUGMENTATION_CONFIG["rotate_90"]:
        k = random.randint(0, 3)
        if k > 0:
            img_arr = np.rot90(img_arr, k).copy()
            mask_arr = np.rot90(mask_arr, k).copy()

    # 4. Mild scale/crop
    crop_range = AUGMENTATION_CONFIG["random_crop_scale"]
    if crop_range[0] < 1.0 and random.random() > 0.5:
        scale = random.uniform(crop_range[0], crop_range[1])
        h, w = img_arr.shape[:2]
        ch, cw = int(h * scale), int(w * scale)
        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)
        img_arr = cv2.resize(img_arr[top:top+ch, left:left+cw], (w, h), interpolation=cv2.INTER_LINEAR)
        mask_arr = cv2.resize(mask_arr[top:top+ch, left:left+cw], (w, h), interpolation=cv2.INTER_NEAREST)

    # 5. Photometric (Image only)
    br = AUGMENTATION_CONFIG["brightness_range"]
    if random.random() > 0.5:
        factor = random.uniform(br[0], br[1])
        img_arr = np.clip(img_arr * factor, 0.0, 1.0)

    cr = AUGMENTATION_CONFIG["contrast_range"]
    if random.random() > 0.5:
        factor = random.uniform(cr[0], cr[1])
        mean_val = img_arr.mean()
        img_arr = np.clip((img_arr - mean_val) * factor + mean_val, 0.0, 1.0)

    # 6. Gaussian noise
    if random.random() < AUGMENTATION_CONFIG["gaussian_noise_prob"]:
        noise = np.random.normal(0, AUGMENTATION_CONFIG["gaussian_noise_std"], img_arr.shape).astype(np.float32)
        img_arr = np.clip(img_arr + noise, 0.0, 1.0)

    return img_arr, mask_arr


class OpenCities3ClassDataset(Dataset):
    """Dataset for 3-Class Open Cities drone segmentation."""
    def __init__(self, split_file: str, is_train: bool = False):
        self.samples = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(",")
                    self.samples.append((parts[0], parts[1]))
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)  # uint8 with values {0, 1, 2}

        img_arr = np.array(img, dtype=np.float32) / 255.0
        mask_arr = np.array(mask, dtype=np.uint8)

        if self.is_train:
            img_arr, mask_arr = apply_augmentations(img_arr, mask_arr)

        # ImageNet normalization
        img_arr = (img_arr - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_arr).long()

        return img_tensor, mask_tensor, os.path.basename(img_path)


# ════════════════════════════════════════════════════════════════════
# Metrics Calculation
# ════════════════════════════════════════════════════════════════════
def compute_batch_metrics(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> dict:
    """Compute building-level (Class 1 or 2) and class-specific metrics."""
    preds = torch.argmax(logits, dim=1)
    
    # Building metrics: Class 1 | Class 2
    p_bldg = (preds == 1) | (preds == 2)
    t_bldg = (targets == 1) | (targets == 2)

    tp_b = (p_bldg & t_bldg).sum().item()
    fp_b = (p_bldg & (~t_bldg)).sum().item()
    fn_b = ((~p_bldg) & t_bldg).sum().item()

    bldg_dice = (2.0 * tp_b + eps) / (2.0 * tp_b + fp_b + fn_b + eps)
    bldg_iou  = (tp_b + eps) / (tp_b + fp_b + fn_b + eps)
    bldg_prec = (tp_b + eps) / (tp_b + fp_b + eps)
    bldg_rec  = (tp_b + eps) / (tp_b + fn_b + eps)

    # Class 1 (interior)
    p_int = (preds == 1)
    t_int = (targets == 1)
    tp_int = (p_int & t_int).sum().item()
    int_dice = (2.0 * tp_int + eps) / (2.0 * tp_int + (p_int & ~t_int).sum().item() + (~p_int & t_int).sum().item() + eps)

    # Class 2 (boundary)
    p_bnd = (preds == 2)
    t_bnd = (targets == 2)
    tp_bnd = (p_bnd & t_bnd).sum().item()
    bnd_dice = (2.0 * tp_bnd + eps) / (2.0 * tp_bnd + (p_bnd & ~t_bnd).sum().item() + (~p_bnd & t_bnd).sum().item() + eps)

    return {
        "bldg_dice": float(bldg_dice),
        "bldg_iou": float(bldg_iou),
        "bldg_precision": float(bldg_prec),
        "bldg_recall": float(bldg_rec),
        "interior_dice": float(int_dice),
        "boundary_dice": float(bnd_dice),
    }


def evaluate_loader(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    """Evaluate full loader and aggregate metrics."""
    model.eval()
    total_loss = 0.0
    accumulated = defaultdict(list)
    n_batches = 0

    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            loss, _, _ = criterion(logits, masks)

            total_loss += loss.item()
            n_batches += 1
            bm = compute_batch_metrics(logits, masks)
            for k, v in bm.items():
                accumulated[k].append(v)

    res = {"loss": total_loss / max(n_batches, 1)}
    for k, v in accumulated.items():
        res[k] = float(np.mean(v))
    return res


# ════════════════════════════════════════════════════════════════════
# Section 10: Sanity Check Before Full Training
# ════════════════════════════════════════════════════════════════════
def run_sanity_check(device: torch.device) -> bool:
    print("\n" + "=" * 60)
    print("PHASE 11: PRE-TRAINING SANITY CHECK (Section 10)")
    print("=" * 60)

    # Build model initialized from SpaceNet checkpoint
    model = get_model("unet", "resnet18", 3, 3, pretrained=False)
    ckpt = torch.load(SPACENET_CKPT, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.train()

    criterion = MulticlassFocalDiceLoss(class_weights=(1.0, 4.0, 3.0), gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    train_file = os.path.join(SPLITS_DIR, "train.txt")
    ds = OpenCities3ClassDataset(train_file, is_train=True)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    print("  Testing 4 batches forward, loss, backward, and optimizer update...")
    losses = []
    for step, (imgs, masks, _) in enumerate(loader):
        if step >= 4:
            break
        imgs = imgs.to(device)
        masks = masks.to(device)

        # 1. Output shape check
        logits = model(imgs)
        assert logits.shape == (4, 3, 256, 256), f"Shape mismatch: {logits.shape}"

        # 2. Labels check
        unique_labels = torch.unique(masks).cpu().numpy().tolist()
        for lbl in unique_labels:
            assert lbl in [0, 1, 2], f"Invalid label {lbl} found!"

        # 3. Finite loss check
        loss, focal, dice = criterion(logits, masks)
        loss_val = loss.item()
        assert not np.isnan(loss_val) and not np.isinf(loss_val), f"Non-finite loss: {loss_val}"
        losses.append(loss_val)

        # 4. Gradient check
        optimizer.zero_grad()
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0 and max(grad_norms) > 0.0, "Gradients are zero!"

        # 5. Optimizer step
        optimizer.step()
        print(f"    Step {step+1}: Loss = {loss_val:.4f} (Focal: {focal.item():.4f}, Dice: {dice.item():.4f})")

    print(f"  Sanity Check Passed! Output shape verified: [4, 3, 256, 256]")
    print(f"  Labels verified: {unique_labels}")
    print(f"  Gradients non-zero, loss finite, parameters updated successfully.")
    print("=" * 60)
    return True


# ════════════════════════════════════════════════════════════════════
# Training Function (Supports SpaceNet Init & ImageNet Ablation)
# ════════════════════════════════════════════════════════════════════
def train_model(
    exp_name: str,
    init_mode: str,  # "spacenet" or "imagenet"
    device: torch.device,
    batch_size: int = 8,
    max_epochs: int = 40,
    patience: int = 7,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> Tuple[str, dict]:
    print("\n" + "─" * 60)
    print(f"STARTING EXPERIMENT: {exp_name}")
    print(f"  Initialization : {init_mode.upper()}")
    print(f"  Max Epochs     : {max_epochs}, Early Stopping Patience: {patience}")
    print(f"  Batch Size     : {batch_size}, LR: {lr}, Weight Decay: {weight_decay}")
    print("─" * 60)

    os.makedirs(CKPT_DIR, exist_ok=True)
    set_seed(42)

    # 1. Dataset & Loaders
    train_ds = OpenCities3ClassDataset(os.path.join(SPLITS_DIR, "train.txt"), is_train=True)
    val_ds   = OpenCities3ClassDataset(os.path.join(SPLITS_DIR, "val.txt"), is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    # 2. Build Model & Initialize
    if init_mode == "spacenet":
        print(f"  Loading pre-trained SpaceNet weights from {SPACENET_CKPT}...")
        model = get_model("unet", "resnet18", 3, 3, pretrained=False)
        ckpt = torch.load(SPACENET_CKPT, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
    elif init_mode == "imagenet":
        print("  Initializing model with ImageNet pre-trained encoder (NO SpaceNet weights)...")
        model = get_model("unet", "resnet18", 3, 3, pretrained=True)
    else:
        raise ValueError(f"Unknown init mode: {init_mode}")

    model.to(device)

    # 3. Loss & Optimizer
    criterion = MulticlassFocalDiceLoss(class_weights=(1.0, 4.0, 3.0), gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # 4. Training Loop
    history = []
    best_val_dice = -1.0
    best_epoch = 0
    patience_counter = 0

    best_ckpt_path = os.path.join(CKPT_DIR, f"{exp_name}_best_model.pth")
    last_ckpt_path = os.path.join(CKPT_DIR, f"{exp_name}_last_model.pth")

    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        t_ep = time.time()
        model.train()
        train_loss = 0.0
        train_acc = defaultdict(list)
        n_train_batches = 0

        for imgs, masks, _ in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss, _, _ = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1
            bm = compute_batch_metrics(logits, masks)
            for k, v in bm.items():
                train_acc[k].append(v)

        train_metrics = {k: float(np.mean(v)) for k, v in train_acc.items()}
        train_loss /= max(n_train_batches, 1)

        # Validation
        val_metrics = evaluate_loader(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        val_dice = val_metrics["bldg_dice"]

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_dice)

        ep_duration = time.time() - t_ep

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_bldg_dice": round(train_metrics["bldg_dice"], 4),
            "train_bldg_iou": round(train_metrics["bldg_iou"], 4),
            "train_bldg_precision": round(train_metrics["bldg_precision"], 4),
            "train_bldg_recall": round(train_metrics["bldg_recall"], 4),
            "val_loss": round(val_loss, 4),
            "val_bldg_dice": round(val_dice, 4),
            "val_bldg_iou": round(val_metrics["bldg_iou"], 4),
            "val_bldg_precision": round(val_metrics["bldg_precision"], 4),
            "val_bldg_recall": round(val_metrics["bldg_recall"], 4),
            "val_interior_dice": round(val_metrics["interior_dice"], 4),
            "val_boundary_dice": round(val_metrics["boundary_dice"], 4),
            "lr": current_lr,
            "epoch_time_s": round(ep_duration, 1),
        }
        history.append(epoch_record)

        print(f"  Ep {epoch:2d}/{max_epochs:2d} [{ep_duration:4.1f}s] | "
              f"Tr Loss: {train_loss:.4f} Tr Dice: {train_metrics['bldg_dice']:.4f} | "
              f"Val Loss: {val_loss:.4f} Val Dice: {val_dice:.4f} (IoU: {val_metrics['bldg_iou']:.4f}) | "
              f"LR: {current_lr:.1e}", end="")

        # Checkpoint saving & Early stopping
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_score": best_val_dice,
                "history": history,
                "init_mode": init_mode,
            }, best_ckpt_path)
            print(f" -> BEST MODEL SAVED (Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1
            print(f" (patience: {patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n  Early stopping triggered at epoch {epoch}! No improvement for {patience} epochs.")
                break

    # Save last model checkpoint
    torch.save({
        "epoch": len(history),
        "model_state_dict": model.state_dict(),
        "last_val_score": history[-1]["val_bldg_dice"],
        "history": history,
    }, last_ckpt_path)

    total_time = time.time() - t_start
    print(f"\nTraining completed in {total_time/60:.1f} minutes. Best Epoch: {best_epoch} with Val Building Dice: {best_val_dice:.4f}")

    # Save CSV history
    csv_file = os.path.join(OUTPUT_DIR, f"{exp_name}_history.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)
    print(f"  Saved history to {csv_file}")

    # Plot training curves
    plot_training_curves(history, exp_name)

    return best_ckpt_path, {
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "total_epochs": len(history),
        "training_time_s": total_time,
        "best_checkpoint": best_ckpt_path,
        "history_csv": csv_file,
    }


def plot_training_curves(history: List[dict], exp_name: str):
    """Plot Loss and Dice curves."""
    epochs = [h["epoch"] for h in history]
    tr_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    tr_dice = [h["train_bldg_dice"] for h in history]
    val_dice = [h["val_bldg_dice"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    plt.subplots_adjust(wspace=0.25)

    # Loss
    axes[0].plot(epochs, tr_loss, "b-o", label="Train Loss", linewidth=2, markersize=4)
    axes[0].plot(epochs, val_loss, "r-s", label="Val Loss", linewidth=2, markersize=4)
    axes[0].set_title(f"{exp_name} — Multiclass Loss", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=9)

    # Dice
    axes[1].plot(epochs, tr_dice, "b-o", label="Train Building Dice", linewidth=2, markersize=4)
    axes[1].plot(epochs, val_dice, "g-^", label="Val Building Dice", linewidth=2, markersize=4)
    axes[1].set_title(f"{exp_name} — Building Dice Curve", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Building Dice")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=9)

    out_png = os.path.join(OUTPUT_DIR, f"{exp_name}_curves.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved training curves to {out_png}")


# ════════════════════════════════════════════════════════════════════
# Main Execution
# ════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Phase 11: Domain Adaptation Training")
    parser.add_argument("--epochs", type=int, default=40, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # 1. Run Sanity Check first
    run_sanity_check(device)

    # 2. Run Primary Experiment: SpaceNet Initialization
    print("\n" + "=" * 70)
    print("PRIMARY EXPERIMENT: DOMAIN ADAPTATION (SPACENET INITIALIZATION)")
    print("=" * 70)
    prim_ckpt, prim_summary = train_model(
        exp_name="training",
        init_mode="spacenet",
        device=device,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
    )

    # Copy checkpoints to primary filenames
    best_target = os.path.join(OUTPUT_DIR, "checkpoints", "best_model.pth")
    last_target = os.path.join(OUTPUT_DIR, "checkpoints", "last_model.pth")
    if os.path.exists(prim_ckpt):
        import shutil
        shutil.copy2(prim_ckpt, best_target)
        last_src = os.path.join(OUTPUT_DIR, "checkpoints", "training_last_model.pth")
        if os.path.exists(last_src):
            shutil.copy2(last_src, last_target)

    # The files training_history.csv and training_curves.png are already saved under OUTPUT_DIR.

    # 3. Run Controlled Ablation: ImageNet Initialization
    print("\n" + "=" * 70)
    print("CONTROLLED ABLATION: IMAGENET INITIALIZATION (NO SPACENET)")
    print("=" * 70)
    abl_ckpt, abl_summary = train_model(
        exp_name="imagenet_ablation",
        init_mode="imagenet",
        device=device,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
    )

    # 4. Save Experiment Configuration
    config_dict = {
        "experiment": "Phase 11: Controlled Open Cities Domain Adaptation",
        "random_seed": 42,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "model_architecture": "U-Net with ResNet-18 Encoder (3 Output Classes)",
        "parameter_count": 14328209,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": 1e-4,
        "batch_size": args.batch_size,
        "maximum_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "loss": "Multiclass Focal (gamma=2.0, weight=0.5) + Multiclass Dice (weight=0.5)",
        "class_weights": [1.0, 4.0, 3.0],
        "augmentation_parameters": AUGMENTATION_CONFIG,
        "spatial_splits": {
            "train": 383,
            "validation": 130,
            "test": 136,
        },
        "primary_experiment": prim_summary,
        "imagenet_ablation": abl_summary,
    }

    config_path = os.path.join(OUTPUT_DIR, "experiment_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    print(f"\nSaved experiment config to {config_path}")

    print("\n" + "=" * 70)
    print("PHASE 11 MODEL TRAINING & ABLATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
