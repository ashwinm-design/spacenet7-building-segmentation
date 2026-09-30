import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.dataset import get_dataloaders
from src.model import get_model, BCEDiceLoss

def compute_metrics(logits, targets, threshold=0.5):
    """
    Computes Intersection over Union (IoU) and Dice coefficient for binary segmentation.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum().item()
    union = preds_flat.sum().item() + targets_flat.sum().item() - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2.0 * intersection + 1e-6) / (preds_flat.sum().item() + targets_flat.sum().item() + 1e-6)

    return iou, dice

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0

    for imgs, masks, _ in dataloader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        iou, dice = compute_metrics(logits, masks)
        running_iou += iou * imgs.size(0)
        running_dice += dice * imgs.size(0)

    total_samples = len(dataloader.dataset)
    return (
        running_loss / total_samples,
        running_iou / total_samples,
        running_dice / total_samples
    )

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for imgs, masks, _ in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            loss = criterion(logits, masks)

            running_loss += loss.item() * imgs.size(0)
            iou, dice = compute_metrics(logits, masks)
            running_iou += iou * imgs.size(0)
            running_dice += dice * imgs.size(0)

    total_samples = len(dataloader.dataset)
    return (
        running_loss / total_samples,
        running_iou / total_samples,
        running_dice / total_samples
    )

def main():
    config_path = "configs/baseline.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Determine device: CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"Starting Training on Device: {device}")
    print(f"==================================================")

    processed_dir = cfg["dataset"]["processed_dir"]
    batch_size = cfg["training"]["batch_size"]
    target_size = tuple(cfg["dataset"]["image_size"])
    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["learning_rate"]
    checkpoint_dir = cfg["training"]["save_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Loading data...")
    loaders = get_dataloaders(processed_dir, batch_size=batch_size, target_size=target_size)
    train_loader = loaders['train']
    val_loader = loaders['val']
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    print("Initializing model...")
    model = get_model(
        architecture=cfg["model"]["architecture"],
        encoder_name=cfg["model"]["encoder_name"],
        in_channels=cfg["model"]["in_channels"],
        classes=cfg["model"]["classes"],
        pretrained=True
    ).to(device)

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_dice = 0.0
    history = []

    print("\nStarting Epochs:")
    print("-" * 75)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val IoU':<10} | {'Val Dice':<10} | {'Time (s)':<8}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_iou, train_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou, val_dice = validate_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"{epoch:<8} | {train_loss:<12.4f} | {val_loss:<10.4f} | {val_iou:<10.4f} | {val_dice:<10.4f} | {elapsed:<8.1f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_dice": val_dice,
            "time_sec": elapsed
        })

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_dice": val_dice,
            "config": cfg
        }

        torch.save(checkpoint, os.path.join(checkpoint_dir, "latest_model.pth"))

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pth"))

    print("-" * 75)
    print(f"Training Complete! Best Validation Dice: {best_val_dice:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")

    # Save training history
    history_file = os.path.join("results/metrics", "training_history.json")
    with open(history_file, "w") as f:
        import json
        json.dump(history, f, indent=2)
    print(f"Training log written to {history_file}")

if __name__ == "__main__":
    main()
