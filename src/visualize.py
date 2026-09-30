import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def create_prediction_visualizations(num_samples=12):
    test_split_path = "data/processed/splits/test.txt"
    pred_dir = "results/predictions"
    vis_dir = "results/visualizations"
    os.makedirs(vis_dir, exist_ok=True)

    with open(test_split_path, "r") as f:
        pairs = [line.strip().split(",") for line in f.readlines() if line.strip()]

    print(f"Creating prediction visualizations for {min(num_samples, len(pairs))} test samples...")

    saved_count = 0
    for idx, (img_path, mask_path) in enumerate(pairs[:num_samples]):
        filename = os.path.basename(img_path)
        pred_filename = f"pred_{filename}"
        pred_path = os.path.join(pred_dir, pred_filename)

        if not os.path.exists(pred_path):
            # check without .png or with .png replacement
            alt_pred_path = os.path.join(pred_dir, f"pred_{os.path.splitext(filename)[0]}.png")
            if os.path.exists(alt_pred_path):
                pred_path = alt_pred_path
            else:
                continue

        # Load RGB image
        orig_img = Image.open(img_path).convert("RGB")
        orig_arr = np.array(orig_img)

        # Load ground truth mask
        gt_mask = Image.open(mask_path).convert("L")
        gt_arr = (np.array(gt_mask) > 127).astype(np.uint8)

        # Load predicted mask
        pred_mask = Image.open(pred_path).convert("L")
        pred_arr = (np.array(pred_mask) > 127).astype(np.uint8)

        # Create overlay:
        # Original image + semi-transparent cyan/green building footprint overlay
        overlay = orig_arr.copy().astype(np.float32)
        # Prediction in cyan/yellow overlay
        color_mask = np.zeros_like(overlay)
        color_mask[pred_arr == 1] = [0, 230, 255]  # Bright Cyan for building footprint prediction

        alpha = 0.45
        mask_indices = pred_arr == 1
        overlay[mask_indices] = (1 - alpha) * overlay[mask_indices] + alpha * color_mask[mask_indices]
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Build 4-panel figure:
        # 1. Original aerial image
        # 2. Ground truth building mask
        # 3. Model prediction
        # 4. Overlay of prediction on the original image
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), dpi=150)

        axes[0].imshow(orig_arr)
        axes[0].set_title(f"1. Aerial Image ({idx+1})", fontsize=11, fontweight="bold", pad=8)
        axes[0].axis("off")

        axes[1].imshow(gt_arr, cmap="gray")
        axes[1].set_title("2. Ground Truth Mask", fontsize=11, fontweight="bold", pad=8)
        axes[1].axis("off")

        axes[2].imshow(pred_arr, cmap="magma")
        axes[2].set_title("3. Predicted Mask (U-Net)", fontsize=11, fontweight="bold", pad=8)
        axes[2].axis("off")

        axes[3].imshow(overlay)
        axes[3].set_title("4. Footprint Overlay", fontsize=11, fontweight="bold", pad=8)
        axes[3].axis("off")

        plt.suptitle(f"Sample {idx+1}: {filename[:32]}...", fontsize=12, y=0.98)
        plt.tight_layout()

        out_path = os.path.join(vis_dir, f"prediction_sample_{idx+1:02d}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        saved_count += 1

    print(f"Successfully generated {saved_count} visualizations in {vis_dir}/")
    return saved_count

if __name__ == "__main__":
    create_prediction_visualizations(12)
