import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import csv
import yaml
import numpy as np
from PIL import Image
import torch

from src.dataset import get_dataloaders
from src.model import get_model

def evaluate_model():
    config_path = "configs/baseline.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"Running Held-Out Test Evaluation on Device: {device}")
    print(f"==================================================")

    processed_dir = cfg["dataset"]["processed_dir"]
    batch_size = 1  # batch size 1 for individual prediction saving
    target_size = tuple(cfg["dataset"]["image_size"])
    checkpoint_path = os.path.join(cfg["training"]["save_dir"], "best_model.pth")

    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(cfg["training"]["save_dir"], "latest_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = get_model(
        architecture=cfg["model"]["architecture"],
        encoder_name=cfg["model"]["encoder_name"],
        in_channels=cfg["model"]["in_channels"],
        classes=cfg["model"]["classes"],
        pretrained=False
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loaders = get_dataloaders(processed_dir, batch_size=batch_size, target_size=target_size)
    test_loader = loaders["test"]
    print(f"Held-out test samples: {len(test_loader.dataset)}")

    pred_dir = "results/predictions"
    os.makedirs(pred_dir, exist_ok=True)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    per_image_metrics = []
    empty_pred_count = 0
    empty_gt_count = 0

    with torch.no_grad():
        for i, (imgs, masks, filenames) in enumerate(test_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            filename = filenames[0]

            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            pred_np = preds.squeeze().cpu().numpy().astype(np.uint8)
            mask_np = masks.squeeze().cpu().numpy().astype(np.uint8)

            # Save prediction mask
            pred_save_path = os.path.join(pred_dir, f"pred_{filename}")
            Image.fromarray((pred_np * 255).astype(np.uint8)).save(pred_save_path)

            tp = np.logical_and(pred_np == 1, mask_np == 1).sum()
            fp = np.logical_and(pred_np == 1, mask_np == 0).sum()
            fn = np.logical_and(pred_np == 0, mask_np == 1).sum()
            tn = np.logical_and(pred_np == 0, mask_np == 0).sum()

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            img_iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
            img_dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
            img_precision = (tp + 1e-6) / (tp + fp + 1e-6)
            img_recall = (tp + 1e-6) / (tp + fn + 1e-6)

            if pred_np.sum() == 0:
                empty_pred_count += 1
            if mask_np.sum() == 0:
                empty_gt_count += 1

            per_image_metrics.append({
                "filename": filename,
                "iou": float(img_iou),
                "dice": float(img_dice),
                "precision": float(img_precision),
                "recall": float(img_recall),
                "pred_building_pixels": int(pred_np.sum()),
                "gt_building_pixels": int(mask_np.sum())
            })

    # Overall dataset metrics
    dataset_iou = (total_tp + 1e-6) / (total_tp + total_fp + total_fn + 1e-6)
    dataset_dice = (2 * total_tp + 1e-6) / (2 * total_tp + total_fp + total_fn + 1e-6)
    dataset_precision = (total_tp + 1e-6) / (total_tp + total_fp + 1e-6)
    dataset_recall = (total_tp + 1e-6) / (total_tp + total_fn + 1e-6)

    test_count = len(test_loader.dataset)
    empty_pred_pct = (empty_pred_count / test_count) * 100.0

    summary_metrics = {
        "dataset": cfg["dataset"]["name"],
        "test_samples": test_count,
        "iou": round(float(dataset_iou), 4),
        "dice": round(float(dataset_dice), 4),
        "precision": round(float(dataset_precision), 4),
        "recall": round(float(dataset_recall), 4),
        "empty_predictions_count": empty_pred_count,
        "empty_predictions_pct": round(empty_pred_pct, 2),
        "total_true_positives": int(total_tp),
        "total_false_positives": int(total_fp),
        "total_false_negatives": int(total_fn),
        "mean_image_iou": round(float(np.mean([m['iou'] for m in per_image_metrics])), 4),
        "mean_image_dice": round(float(np.mean([m['dice'] for m in per_image_metrics])), 4)
    }

    # Save to JSON
    json_path = "results/metrics/baseline_metrics.json"
    with open(json_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"Saved metrics JSON to: {json_path}")

    # Save per-image details to CSV
    csv_path = "results/metrics/baseline_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "iou", "dice", "precision", "recall", "pred_pixels", "gt_pixels"])
        for row in per_image_metrics:
            writer.writerow([
                row["filename"],
                f"{row['iou']:.4f}",
                f"{row['dice']:.4f}",
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                row["pred_building_pixels"],
                row["gt_building_pixels"]
            ])
        writer.writerow([])
        writer.writerow(["OVERALL_DATASET", f"{dataset_iou:.4f}", f"{dataset_dice:.4f}", f"{dataset_precision:.4f}", f"{dataset_recall:.4f}", total_tp + total_fp, total_tp + total_fn])
    print(f"Saved metrics CSV to: {csv_path}")

    print("\nHeld-Out Test Results:")
    print("-" * 40)
    print(f"IoU:       {dataset_iou:.4f}")
    print(f"Dice (F1): {dataset_dice:.4f}")
    print(f"Precision: {dataset_precision:.4f}")
    print(f"Recall:    {dataset_recall:.4f}")
    print(f"Empty Preds: {empty_pred_count}/{test_count} ({empty_pred_pct:.1f}%)")
    print("-" * 40)

    return summary_metrics

if __name__ == "__main__":
    evaluate_model()
