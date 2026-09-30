"""
Phase 1: Tiny Data Overfit Sanity Test (16 samples).

Validates whether the U-Net model can strongly overfit and memorize
16 SpaceNet 7 training image-mask pairs with augmentations disabled.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import csv
import json
import argparse
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import BuildingDataset
from src.model import get_model, BCEDiceLoss


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6):
    """
    Compute Dice, IoU, Precision, and Recall on batch tensors.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum().item()
    total_pred = preds_flat.sum().item()
    total_target = targets_flat.sum().item()
    union = total_pred + total_target - intersection

    dice = (2.0 * intersection + eps) / (total_pred + total_target + eps)
    iou = (intersection + eps) / (union + eps)
    precision = (intersection + eps) / (total_pred + eps)
    recall = (intersection + eps) / (total_target + eps)

    return dice, iou, precision, recall


def save_prediction_visualizations(
    model: nn.Module,
    samples: list,
    output_dir: str,
    device: torch.device
):
    """
    Save prediction artifacts for samples:
    1. Original image
    2. Ground-truth mask
    3. Predicted probability mask
    4. Thresholded predicted mask
    5. Side-by-side composite panel
    """
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    model.eval()

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    saved_records = []

    with torch.no_grad():
        for item in samples:
            img_path = item["img_path"]
            mask_path = item["mask_path"]
            name = item["sample_name"]

            # Load raw RGB image & ground truth
            raw_rgb = Image.open(img_path).convert("RGB")
            raw_mask = Image.open(mask_path).convert("L")

            # Prepare tensor for model
            img_arr = np.array(raw_rgb, dtype=np.float32) / 255.0
            norm_img = (img_arr - mean) / std
            tensor_x = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

            # Predict
            logits = model(tensor_x)
            prob_tensor = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_binary = (prob_tensor > 0.5).astype(np.uint8) * 255
            prob_uint8 = np.clip(prob_tensor * 255, 0, 255).astype(np.uint8)

            # GT mask uint8
            gt_arr = (np.array(raw_mask) > 127).astype(np.uint8) * 255

            # Save individual files
            orig_file = os.path.join(pred_dir, f"{name}_original.png")
            gt_file = os.path.join(pred_dir, f"{name}_gt.png")
            prob_file = os.path.join(pred_dir, f"{name}_prob.png")
            pred_file = os.path.join(pred_dir, f"{name}_pred.png")
            comp_file = os.path.join(pred_dir, f"{name}_comparison.png")

            raw_rgb.save(orig_file)
            Image.fromarray(gt_arr).save(gt_file)
            # Save probability map as colormap
            prob_color = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_MAGMA)
            prob_color_rgb = cv2.cvtColor(prob_color, cv2.COLOR_BGR2RGB)
            Image.fromarray(prob_color_rgb).save(prob_file)
            Image.fromarray(pred_binary).save(pred_file)

            # Compose side-by-side comparison
            # Original RGB | GT Mask | Pred Mask | Probability Heatmap
            w, h = 256, 256
            header_h = 32
            comp_w = w * 4
            comp_h = h + header_h
            composite = np.zeros((comp_h, comp_w, 3), dtype=np.uint8)
            composite[:header_h, :] = [24, 28, 36]

            gt_rgb = cv2.cvtColor(gt_arr, cv2.COLOR_GRAY2RGB)
            pred_rgb = cv2.cvtColor(pred_binary, cv2.COLOR_GRAY2RGB)

            composite[header_h:, 0:w] = np.array(raw_rgb)
            composite[header_h:, w:w*2] = gt_rgb
            composite[header_h:, w*2:w*3] = pred_rgb
            composite[header_h:, w*3:w*4] = prob_color_rgb

            # Headers
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            cv2.putText(composite, "1. Aerial Image", (10, 21), font, font_scale, (220, 225, 235), 1, cv2.LINE_AA)
            cv2.putText(composite, "2. Ground Truth", (w + 10, 21), font, font_scale, (220, 225, 235), 1, cv2.LINE_AA)
            cv2.putText(composite, "3. Predicted Mask", (w * 2 + 10, 21), font, font_scale, (80, 240, 140), 1, cv2.LINE_AA)
            cv2.putText(composite, "4. Probability Map", (w * 3 + 10, 21), font, font_scale, (240, 180, 80), 1, cv2.LINE_AA)

            # Grid lines
            for i in range(1, 4):
                cv2.line(composite, (w * i, 0), (w * i, comp_h), (50, 58, 70), 1)

            Image.fromarray(composite).save(comp_file)
            saved_records.append({
                "sample_name": name,
                "comparison": comp_file,
                "gt_pixels": int(np.sum(gt_arr > 0)),
                "pred_pixels": int(np.sum(pred_binary > 0))
            })

    return saved_records


def plot_training_curves(metrics_csv: str, output_png: str):
    """Plot multi-panel training diagnostic curves."""
    epochs, losses, dices, ious, precisions, recalls = [], [], [], [], [], []

    with open(metrics_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))
            dices.append(float(row["dice"]))
            ious.append(float(row["iou"]))
            precisions.append(float(row["precision"]))
            recalls.append(float(row["recall"]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0b0f19")

    for ax in axes:
        ax.set_facecolor("#141c2e")
        ax.tick_params(colors="#94a3b8")
        ax.spines["bottom"].set_color("#23304a")
        ax.spines["top"].set_color("#23304a")
        ax.spines["left"].set_color("#23304a")
        ax.spines["right"].set_color("#23304a")
        ax.grid(True, color="#23304a", linestyle="--", alpha=0.6)

    # Panel 1: Loss
    axes[0].plot(epochs, losses, color="#ef4444", linewidth=2.2, label="BCE + Dice Loss")
    axes[0].set_title("Training Loss Overfit Curve", color="#fff", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", color="#94a3b8")
    axes[0].set_ylabel("Loss", color="#94a3b8")
    axes[0].legend(facecolor="#141c2e", edgecolor="#23304a", labelcolor="#fff")

    # Panel 2: Dice & IoU
    axes[1].plot(epochs, dices, color="#10b981", linewidth=2.2, label="Dice Score")
    axes[1].plot(epochs, ious, color="#3b82f6", linewidth=2.2, linestyle="--", label="IoU (Jaccard)")
    axes[1].set_title("Overlap Metrics (Dice / IoU)", color="#fff", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", color="#94a3b8")
    axes[1].set_ylabel("Score", color="#94a3b8")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(facecolor="#141c2e", edgecolor="#23304a", labelcolor="#fff")

    # Panel 3: Precision & Recall
    axes[2].plot(epochs, precisions, color="#f59e0b", linewidth=2.0, label="Precision")
    axes[2].plot(epochs, recalls, color="#a855f7", linewidth=2.0, linestyle="--", label="Recall")
    axes[2].set_title("Precision & Recall", color="#fff", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch", color="#94a3b8")
    axes[2].set_ylabel("Score", color="#94a3b8")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].legend(facecolor="#141c2e", edgecolor="#23304a", labelcolor="#fff")

    plt.tight_layout()
    plt.savefig(output_png, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def run_overfit_experiment(
    num_samples: int = 16,
    num_epochs: int = 35,
    batch_size: int = 8,
    lr: float = 1e-3,
    output_dir: str = "results/experiments/overfit_16"
):
    """Execute the controlled 16-sample overfit sanity experiment."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("PHASE 1: SPACE NET 7 TINY DATA OVERFIT SANITY TEST")
    print("=" * 65)
    print(f"Device:            {device}")
    print(f"Training Samples:  {num_samples}")
    print(f"Target Epochs:     {num_epochs}")
    print(f"Batch Size:        {batch_size}")
    print(f"Initial LR:        {lr}")
    print(f"Augmentation:      DISABLED (Controlled memorization test)")
    print("-" * 65)

    # 1. Load the first 16 training pairs from SpaceNet 7
    train_split_file = "data/processed_spacenet7/splits/train.txt"
    if not os.path.exists(train_split_file):
        raise FileNotFoundError(f"Split file missing: {train_split_file}")

    with open(train_split_file, "r", encoding="utf-8") as f:
        lines = [line.strip().split(",") for line in f.readlines() if line.strip()]

    selected_pairs = lines[:num_samples]
    sample_records = []
    for img_p, mask_p in selected_pairs:
        sample_records.append({
            "img_path": img_p,
            "mask_path": mask_p,
            "sample_name": os.path.splitext(os.path.basename(img_p))[0]
        })

    img_paths = [r["img_path"] for r in sample_records]
    mask_paths = [r["mask_path"] for r in sample_records]

    # Dataset with augmentations disabled (is_train=False)
    dataset = BuildingDataset(img_paths, mask_paths, target_size=(256, 256), is_train=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Build Model
    model = get_model(architecture="unet", encoder_name="resnet18", in_channels=3, classes=1, pretrained=True)
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = "U-Net (ResNet18 Pretrained Backbone)"

    print(f"Model Architecture: {model_name}")
    print(f"Trainable Params:   {param_count:,}")
    print("-" * 65)

    # 3. Loss & Optimizer
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 4. Training Loop
    metrics_csv_path = os.path.join(output_dir, "metrics.csv")
    csv_file = open(metrics_csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["epoch", "loss", "dice", "iou", "precision", "recall", "lr", "epoch_time_s"])

    best_dice = -1.0
    best_model_path = os.path.join(output_dir, "best_model.pth")

    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_t0 = time.time()
        model.train()

        total_loss = 0.0
        all_logits = []
        all_targets = []

        for imgs, masks, _ in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            all_logits.append(logits.detach())
            all_targets.append(masks.detach())

        avg_loss = total_loss / len(dataset)
        cat_logits = torch.cat(all_logits, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)

        epoch_dice, epoch_iou, epoch_prec, epoch_rec = compute_metrics(cat_logits, cat_targets)
        epoch_time = time.time() - epoch_t0

        writer.writerow([
            epoch,
            f"{avg_loss:.5f}",
            f"{epoch_dice:.4f}",
            f"{epoch_iou:.4f}",
            f"{epoch_prec:.4f}",
            f"{epoch_rec:.4f}",
            f"{lr:.6f}",
            f"{epoch_time:.2f}"
        ])
        csv_file.flush()

        if epoch_dice > best_dice:
            best_dice = epoch_dice
            torch.save(model.state_dict(), best_model_path)

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"Epoch [{epoch:2d}/{num_epochs:2d}] | Loss: {avg_loss:.4f} | Dice: {epoch_dice:.4f} | IoU: {epoch_iou:.4f} | Prec: {epoch_prec:.4f} | Rec: {epoch_rec:.4f} ({epoch_time:.1f}s)")

    csv_file.close()
    total_training_time = time.time() - t_start

    print("-" * 65)
    print(f"Training completed in {total_training_time:.1f}s")
    print(f"Best Dice achieved: {best_dice:.4f} (Model saved to {best_model_path})")

    # 5. Plot training curves
    curve_png_path = os.path.join(output_dir, "training_curve.png")
    plot_training_curves(metrics_csv_path, curve_png_path)
    print(f"Training curves saved to: {curve_png_path}")

    # 6. Load best model and generate predictions for all 16 samples
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    saved_preds = save_prediction_visualizations(model, sample_records, output_dir, device)
    print(f"Saved {len(saved_preds)} prediction visualizations to: {os.path.join(output_dir, 'predictions')}")

    # Determine success threshold
    overfit_success = best_dice >= 0.85
    status_str = "SUCCESS" if overfit_success else "FAILED (Insufficient Memorization)"

    summary = {
        "model_architecture": model_name,
        "parameter_count": param_count,
        "num_epochs": num_epochs,
        "final_loss": round(avg_loss, 5),
        "final_dice": round(epoch_dice, 4),
        "best_dice": round(best_dice, 4),
        "final_iou": round(epoch_iou, 4),
        "final_precision": round(epoch_prec, 4),
        "final_recall": round(epoch_rec, 4),
        "overfit_test_succeeded": overfit_success,
        "total_training_time_s": round(total_training_time, 2),
        "saved_predictions_dir": os.path.join(output_dir, "predictions"),
        "best_model_path": best_model_path,
        "training_curve_path": curve_png_path,
        "metrics_csv_path": metrics_csv_path
    }

    summary_json_path = os.path.join(output_dir, "overfit_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 65)
    print(f"OVERFIT SANITY TEST RESULT: [{status_str}]")
    print(f"  - Final Loss:       {summary['final_loss']}")
    print(f"  - Best Dice:        {summary['best_dice']}")
    print(f"  - Final IoU:        {summary['final_iou']}")
    print(f"  - Precision:        {summary['final_precision']}")
    print(f"  - Recall:           {summary['final_recall']}")
    print("=" * 65)

    return summary


def main():
    parser = argparse.ArgumentParser(description="SpaceNet 7 Tiny Data Overfit Sanity Test")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of training samples (default 16)")
    parser.add_argument("--epochs", type=int, default=35, help="Number of epochs (default 35)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default 8)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default 1e-3)")
    parser.add_argument("--output-dir", default="results/experiments/overfit_16", help="Results directory")
    args = parser.parse_args()

    run_overfit_experiment(
        num_samples=args.num_samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
