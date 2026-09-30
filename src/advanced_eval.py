"""
Advanced evaluation module for SIH26012 building segmentation baseline.

Includes:
  C. Building-size analysis (small / medium / large)
  D. Boundary quality metrics (boundary F1, Hausdorff distance)
  F. Best/worst prediction gallery ranked by IoU
  J. Visual failure analysis
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import csv
import yaml
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import torch

from src.dataset import get_dataloaders
from src.model import get_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_boundary(mask, width=2):
    """Extract boundary pixels from a binary mask using morphological ops."""
    dilated = binary_dilation(mask, iterations=width)
    eroded = binary_erosion(mask, iterations=width)
    return (dilated.astype(np.uint8) - eroded.astype(np.uint8)).clip(0, 1)


def boundary_f1(pred_mask, gt_mask, boundary_width=2):
    """Boundary F1 score: precision/recall of boundary pixels."""
    pred_bd = extract_boundary(pred_mask, boundary_width)
    gt_bd = extract_boundary(gt_mask, boundary_width)

    tp = np.logical_and(pred_bd == 1, gt_bd == 1).sum()
    fp = np.logical_and(pred_bd == 1, gt_bd == 0).sum()
    fn = np.logical_and(pred_bd == 0, gt_bd == 1).sum()

    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return float(f1), float(precision), float(recall)


def hausdorff_distance(pred_mask, gt_mask, subsample=500):
    """
    Approximate Hausdorff distance between boundary point-sets.
    Sub-samples points for computational efficiency on CPU.
    """
    pred_bd = extract_boundary(pred_mask, 1)
    gt_bd = extract_boundary(gt_mask, 1)

    pred_pts = np.argwhere(pred_bd > 0)
    gt_pts = np.argwhere(gt_bd > 0)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("nan")

    # Subsample for speed
    if len(pred_pts) > subsample:
        idx = np.random.choice(len(pred_pts), subsample, replace=False)
        pred_pts = pred_pts[idx]
    if len(gt_pts) > subsample:
        idx = np.random.choice(len(gt_pts), subsample, replace=False)
        gt_pts = gt_pts[idx]

    d_forward = directed_hausdorff(pred_pts, gt_pts)[0]
    d_backward = directed_hausdorff(gt_pts, pred_pts)[0]
    return float(max(d_forward, d_backward))


def label_connected_components(mask):
    """Return labelled array and component properties (area in pixels)."""
    from scipy.ndimage import label as scipy_label
    labelled, num_features = scipy_label(mask)
    areas = []
    for i in range(1, num_features + 1):
        areas.append(int((labelled == i).sum()))
    return labelled, areas


def categorize_building_size(area_px, small_thresh=200, large_thresh=1500):
    """Classify a building by pixel area into small / medium / large."""
    if area_px < small_thresh:
        return "small"
    elif area_px < large_thresh:
        return "medium"
    else:
        return "large"


# ---------------------------------------------------------------------------
# Main advanced evaluation
# ---------------------------------------------------------------------------

def run_advanced_evaluation():
    config_path = "configs/baseline.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("ADVANCED EVALUATION — SIH26012 Building Segmentation")
    print(f"Device: {device}")
    print("=" * 60)

    processed_dir = cfg["dataset"]["processed_dir"]
    target_size = tuple(cfg["dataset"]["image_size"])
    checkpoint_dir = cfg["training"]["save_dir"]

    # Load best checkpoint
    best_ckpt = os.path.join(checkpoint_dir, "best_model.pth")
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(checkpoint_dir, "latest_model.pth")
    checkpoint = torch.load(best_ckpt, map_location=device)

    model = get_model(
        architecture=cfg["model"]["architecture"],
        encoder_name=cfg["model"]["encoder_name"],
        in_channels=cfg["model"]["in_channels"],
        classes=cfg["model"]["classes"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test split paths
    with open(os.path.join(processed_dir, "splits", "test.txt"), "r") as f:
        pairs = [line.strip().split(",") for line in f if line.strip()]

    loaders = get_dataloaders(processed_dir, batch_size=1, target_size=target_size)
    test_loader = loaders["test"]

    # ── Per-image evaluation ──────────────────────────────────────────────
    per_image = []
    size_metrics = {"small": [], "medium": [], "large": []}
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for idx, (imgs, masks, filenames) in enumerate(test_loader):
            imgs_dev = imgs.to(device)
            logits = model(imgs_dev)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred = (probs > 0.5).astype(np.uint8)
            gt = masks.squeeze().cpu().numpy().astype(np.uint8)

            # Pixel metrics
            tp = np.logical_and(pred == 1, gt == 1).sum()
            fp = np.logical_and(pred == 1, gt == 0).sum()
            fn = np.logical_and(pred == 0, gt == 1).sum()
            iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
            dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
            prec = (tp + 1e-6) / (tp + fp + 1e-6)
            rec = (tp + 1e-6) / (tp + fn + 1e-6)

            # Boundary quality (D)
            bf1, bprec, brec = boundary_f1(pred, gt)
            hd = hausdorff_distance(pred, gt)

            # Unnormalise image for visualisation later
            img_np = imgs.squeeze().permute(1, 2, 0).numpy()
            img_vis = np.clip(img_np * STD + MEAN, 0, 1)

            record = {
                "idx": idx,
                "filename": filenames[0],
                "iou": float(iou),
                "dice": float(dice),
                "precision": float(prec),
                "recall": float(rec),
                "boundary_f1": float(bf1),
                "boundary_precision": float(bprec),
                "boundary_recall": float(brec),
                "hausdorff_distance": float(hd) if not np.isnan(hd) else None,
                "gt_building_px": int(gt.sum()),
                "pred_building_px": int(pred.sum()),
            }
            per_image.append(record)

            # ── C: Building-size analysis ────────────────────────────────
            _, gt_areas = label_connected_components(gt)
            for area in gt_areas:
                cat = categorize_building_size(area)
                size_metrics[cat].append({
                    "image_idx": idx,
                    "area_px": area,
                })

            # Stash arrays for gallery rendering
            record["_img_vis"] = img_vis
            record["_gt"] = gt
            record["_pred"] = pred

    # ── C: Per-size-category IoU ──────────────────────────────────────────
    print("\n── C: Building-size analysis ──")
    size_summary = {}
    for cat in ["small", "medium", "large"]:
        count = len(size_metrics[cat])
        avg_area = np.mean([b["area_px"] for b in size_metrics[cat]]) if count else 0
        # Aggregate IoU from images containing that category
        involved_images = set(b["image_idx"] for b in size_metrics[cat])
        ious = [per_image[i]["iou"] for i in involved_images]
        dices = [per_image[i]["dice"] for i in involved_images]
        recalls = [per_image[i]["recall"] for i in involved_images]
        size_summary[cat] = {
            "building_count": count,
            "mean_area_px": round(avg_area, 1),
            "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
            "mean_dice": round(float(np.mean(dices)), 4) if dices else None,
            "mean_recall": round(float(np.mean(recalls)), 4) if recalls else None,
        }
        print(f"  {cat:>7}: {count:3d} buildings | avg area {avg_area:6.0f} px | "
              f"IoU {size_summary[cat]['mean_iou']:.4f} | "
              f"Dice {size_summary[cat]['mean_dice']:.4f} | "
              f"Recall {size_summary[cat]['mean_recall']:.4f}")

    # ── D: Boundary quality summary ───────────────────────────────────────
    print("\n── D: Boundary quality ──")
    mean_bf1 = np.mean([r["boundary_f1"] for r in per_image])
    valid_hd = [r["hausdorff_distance"] for r in per_image if r["hausdorff_distance"] is not None]
    mean_hd = np.mean(valid_hd) if valid_hd else float("nan")
    print(f"  Mean Boundary F1:          {mean_bf1:.4f}")
    print(f"  Mean Hausdorff Distance:   {mean_hd:.2f} px")

    # ── F: Best / worst gallery ───────────────────────────────────────────
    print("\n── F: Generating best/worst prediction galleries ──")
    sorted_by_iou = sorted(per_image, key=lambda r: r["iou"], reverse=True)
    best_10 = sorted_by_iou[:10]
    worst_10 = sorted_by_iou[-10:][::-1]  # ascending (worst first)

    gallery_dir = "results/visualizations"
    os.makedirs(gallery_dir, exist_ok=True)

    def save_gallery(records, prefix):
        for rank, rec in enumerate(records):
            fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=130)
            axes[0].imshow(rec["_img_vis"])
            axes[0].set_title("Aerial Image", fontsize=10, fontweight="bold")
            axes[0].axis("off")

            axes[1].imshow(rec["_gt"], cmap="gray")
            axes[1].set_title("Ground Truth", fontsize=10, fontweight="bold")
            axes[1].axis("off")

            axes[2].imshow(rec["_pred"], cmap="magma")
            axes[2].set_title("Prediction", fontsize=10, fontweight="bold")
            axes[2].axis("off")

            overlay = rec["_img_vis"].copy()
            color = np.zeros_like(overlay)
            color[rec["_pred"] == 1] = [0, 0.9, 1.0]
            alpha = 0.45
            m = rec["_pred"] == 1
            overlay[m] = (1 - alpha) * overlay[m] + alpha * color[m]
            axes[3].imshow(np.clip(overlay, 0, 1))
            axes[3].set_title("Overlay", fontsize=10, fontweight="bold")
            axes[3].axis("off")

            plt.suptitle(
                f"{prefix.upper()} #{rank+1}  IoU={rec['iou']:.3f}  Dice={rec['dice']:.3f}  "
                f"BF1={rec['boundary_f1']:.3f}   {rec['filename'][:30]}",
                fontsize=10, y=0.99,
            )
            plt.tight_layout()
            plt.savefig(os.path.join(gallery_dir, f"{prefix}_{rank+1:02d}.png"), bbox_inches="tight")
            plt.close()

    save_gallery(best_10, "best")
    save_gallery(worst_10, "worst")
    print(f"  Saved 10 best + 10 worst to {gallery_dir}/")

    # ── J: Visual failure analysis ────────────────────────────────────────
    print("\n── J: Visual failure analysis ──")
    failure_notes = []
    for rec in worst_10:
        notes = []
        if rec["pred_building_px"] == 0:
            notes.append("empty_prediction")
        if rec["recall"] < 0.3:
            notes.append("missed_buildings")
        if rec["precision"] < 0.3:
            notes.append("high_false_positives")
        if rec["boundary_f1"] < 0.2:
            notes.append("poor_boundary")
        if rec["hausdorff_distance"] is not None and rec["hausdorff_distance"] > 30:
            notes.append("large_shape_error")
        failure_notes.append({
            "filename": rec["filename"],
            "iou": rec["iou"],
            "notes": notes if notes else ["general_low_iou"],
        })
        print(f"  {rec['filename'][:35]:35s}  IoU={rec['iou']:.3f}  → {', '.join(notes) if notes else 'general_low_iou'}")

    # ── Save everything ───────────────────────────────────────────────────
    # Strip numpy arrays before serialising
    serialisable = []
    for r in per_image:
        sr = {k: v for k, v in r.items() if not k.startswith("_")}
        serialisable.append(sr)

    out = {
        "per_image_metrics": serialisable,
        "building_size_analysis": size_summary,
        "boundary_quality": {
            "mean_boundary_f1": round(mean_bf1, 4),
            "mean_hausdorff_distance_px": round(mean_hd, 2) if not np.isnan(mean_hd) else None,
        },
        "failure_analysis": failure_notes,
    }

    metrics_dir = "results/metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, "advanced_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved advanced_metrics.json to {metrics_dir}/")

    # CSV for building-size table
    with open(os.path.join(metrics_dir, "building_size_analysis.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "count", "mean_area_px", "mean_iou", "mean_dice", "mean_recall"])
        for cat in ["small", "medium", "large"]:
            s = size_summary[cat]
            w.writerow([cat, s["building_count"], s["mean_area_px"], s["mean_iou"], s["mean_dice"], s["mean_recall"]])

    return out


if __name__ == "__main__":
    run_advanced_evaluation()
