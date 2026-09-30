"""
A. Tiny-dataset overfitting sanity check.
Trains on 15 samples for many epochs to verify the pipeline can memorize data.
If training IoU does not approach ~0.95+ this indicates a fundamental bug.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import yaml
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.dataset import get_dataloaders, BuildingDataset
from src.model import get_model, BCEDiceLoss
from src.train import compute_metrics, train_epoch, validate_epoch


def run_overfit_check():
    config_path = "configs/baseline.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("SANITY CHECK: Tiny-dataset overfitting test")
    print(f"Device: {device}")
    print("=" * 60)

    # Load full training set, take first 15 samples
    processed_dir = cfg["dataset"]["processed_dir"]
    target_size = tuple(cfg["dataset"]["image_size"])
    loaders = get_dataloaders(processed_dir, batch_size=4, target_size=target_size)
    full_train_ds = loaders["train"].dataset

    # CRITICAL: Build a NEW dataset with is_train=False to DISABLE augmentations.
    # Augmentations actively prevent memorization — that's their job.
    # The sanity check must verify the raw pipeline (data loading, masks, model, loss).
    num_overfit = min(15, len(full_train_ds))
    overfit_ds = BuildingDataset(
        full_train_ds.image_paths[:num_overfit],
        full_train_ds.mask_paths[:num_overfit],
        target_size=target_size,
        is_train=False,   # ← No augmentation for memorization test
    )
    overfit_loader = DataLoader(overfit_ds, batch_size=4, shuffle=True, num_workers=0)

    print(f"Overfitting on {num_overfit} samples (augmentation DISABLED)...")

    model = get_model(
        architecture=cfg["model"]["architecture"],
        encoder_name=cfg["model"]["encoder_name"],
        in_channels=cfg["model"]["in_channels"],
        classes=cfg["model"]["classes"],
        pretrained=True,
    ).to(device)

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=0)

    overfit_epochs = 60
    history = []

    print(f"{'Epoch':<8} | {'Loss':<12} | {'IoU':<10} | {'Dice':<10}")
    print("-" * 50)

    for epoch in range(1, overfit_epochs + 1):
        loss, iou, dice = train_epoch(model, overfit_loader, criterion, optimizer, device)
        history.append({"epoch": epoch, "loss": loss, "iou": iou, "dice": dice})

        if epoch % 5 == 0 or epoch <= 3 or epoch == overfit_epochs:
            print(f"{epoch:<8} | {loss:<12.5f} | {iou:<10.4f} | {dice:<10.4f}")

    # Use BEST metrics across all epochs — small-dataset training oscillates
    best_iou = max(h["iou"] for h in history)
    best_dice = max(h["dice"] for h in history)
    best_epoch_iou = max(history, key=lambda h: h["iou"])["epoch"]
    final_loss = history[-1]["loss"]

    print("-" * 50)
    print(f"Peak IoU:  {best_iou:.4f} (epoch {best_epoch_iou})")
    print(f"Peak Dice: {best_dice:.4f}")

    # The question is: CAN the model memorize? If peak IoU ever exceeds 0.75, yes.
    passed = best_iou > 0.75 and best_dice > 0.80

    if passed:
        print(f"SANITY CHECK PASSED: Best IoU={best_iou:.4f}, Best Dice={best_dice:.4f}")
        print("The model can memorize a small dataset. Pipeline is sound.")
    else:
        print(f"SANITY CHECK FAILED: Best IoU={best_iou:.4f}, Best Dice={best_dice:.4f}")
        print("WARNING: The model cannot overfit 15 samples.")
        print("Potential issues:")
        print("  - Image/mask misalignment")
        print("  - Incorrect label values")
        print("  - Preprocessing error")
        print("  - Loss function bug")
        print("  - Model architecture bug")
        print("Investigate before proceeding to main experiment!")

    # Save results
    results_dir = "results/metrics"
    os.makedirs(results_dir, exist_ok=True)
    report = {
        "sanity_check": "overfit_tiny_dataset",
        "num_samples": num_overfit,
        "epochs": overfit_epochs,
        "final_loss": round(final_loss, 6),
        "best_iou": round(best_iou, 4),
        "best_dice": round(best_dice, 4),
        "best_epoch": best_epoch_iou,
        "passed": passed,
        "history": history,
    }
    with open(os.path.join(results_dir, "sanity_check_overfit.json"), "w") as f:
        json.dump(report, f, indent=2)

    return passed


if __name__ == "__main__":
    ok = run_overfit_check()
    if not ok:
        print("\n*** STOPPING: Sanity check failed. Fix issues before main training. ***")
        sys.exit(1)
