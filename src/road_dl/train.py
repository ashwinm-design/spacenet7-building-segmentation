"""
src/road_dl/train.py

Training engine for deep-learning road extraction model.
Executes supervised training using compound BCE + Dice loss, AdamW optimizer,
learning rate scheduling, deterministic random seeding, and checkpoint persistence.
"""

import time
import random
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import get_road_model
from .losses import BCEDiceLoss
from .dataset import RoadPatchDataset
from .evaluation import compute_road_metrics


def train_road_model(
    patch_dir: Path,
    split_info: Dict[str, Any],
    checkpoint_dir: Path,
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Executes training loop and records epoch-by-epoch loss and validation metrics.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_ids = split_info["train_patches"]
    val_ids = split_info["val_patches"]

    train_ds = RoadPatchDataset(patch_dir, train_ids, augment=True)
    val_ds = RoadPatchDataset(patch_dir, val_ids, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = get_road_model(encoder_name="resnet18", classes=1, pretrained=True)
    model.to(device)

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: List[Dict[str, Any]] = []
    best_val_dice = -1.0
    best_epoch = 0
    best_checkpoint_path = checkpoint_dir / "best_road_model.pth"

    print(f"Starting training on {len(train_ids)} train patches, {len(val_ids)} val patches ({device})...")

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss = 0.0
        train_preds_list = []
        train_targets_list = []

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(imgs)
            probs = torch.sigmoid(logits)
            train_preds_list.append((probs > 0.5).cpu().numpy().astype(np.uint8))
            train_targets_list.append(masks.cpu().numpy().astype(np.uint8))

        scheduler.step()
        train_loss /= len(train_ds)
        train_preds = np.concatenate(train_preds_list, axis=0)
        train_targets = np.concatenate(train_targets_list, axis=0)
        train_metrics = compute_road_metrics(train_preds, train_targets)

        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item() * len(imgs)

                probs = torch.sigmoid(logits)
                val_preds_list.append((probs > 0.5).cpu().numpy().astype(np.uint8))
                val_targets_list.append(masks.cpu().numpy().astype(np.uint8))

        val_loss /= len(val_ds)
        val_preds = np.concatenate(val_preds_list, axis=0)
        val_targets = np.concatenate(val_targets_list, axis=0)
        val_metrics = compute_road_metrics(val_preds, val_targets)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_iou": train_metrics["iou"],
            "train_dice": train_metrics["dice"],
            "val_loss": round(val_loss, 4),
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "lr": round(optimizer.param_groups[0]["lr"], 6),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f}, IoU: {train_metrics['iou']:.4f}, Dice: {train_metrics['dice']:.4f} | "
            f"Val Loss: {val_loss:.4f}, IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": best_val_dice,
                "val_iou": val_metrics["iou"],
                "val_metrics": val_metrics,
            }, best_checkpoint_path)

    total_time = round(time.time() - t_start, 2)
    print(f"\nTraining completed in {total_time}s. Best Epoch: {best_epoch} with Val Dice: {best_val_dice:.4f}")

    # Load best checkpoint into model
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    summary = {
        "total_training_time_seconds": total_time,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "best_val_iou": checkpoint["val_iou"],
        "best_val_metrics": checkpoint["val_metrics"],
        "checkpoint_path": str(best_checkpoint_path),
        "history": history,
    }

    return model, summary
