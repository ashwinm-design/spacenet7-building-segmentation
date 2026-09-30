"""
Phase 5: 3-Class Building Segmentation Visual Validation Tool.

Generates high-resolution multi-panel visualizations for at least 12 representative patches
covering all 7 required structural categories:
1. isolated_buildings
2. dense_buildings
3. touching_adjacent
4. small_buildings
5. large_buildings
6. irregular_buildings
7. empty_background

Each visualization contains:
  1. Original aerial RGB image
  2. Binary building mask (ground truth baseline)
  3. 3-class target mask (Class 0: Background, Class 1: Interior, Class 2: Boundary)
  4. Zoomed view focusing specifically on dense / touching building boundaries

Also produces a grand composite overview grid (00_master_3class_overview.png)
and copies visualizations to the artifacts directory for reporting.
"""

import os
import sys
import json
import shutil
from typing import Dict, List, Tuple, Any

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

# Ensure standard output uses UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Curated palette for 3-class target:
# Class 0: Dark Charcoal [30, 32, 40]
# Class 1: Vivid Emerald Green [46, 204, 113]
# Class 2: Vivid Coral Red [231, 76, 60]
COLOR_PALETTE = np.array([
    [30, 32, 40],      # Class 0: Background
    [46, 204, 113],    # Class 1: Building Interior
    [231, 76, 60],     # Class 2: Building Boundary
], dtype=np.uint8)

CMAP_3CLASS = ListedColormap([
    "#1e2028",  # 0: Background (Dark charcoal)
    "#2ecc71",  # 1: Interior (Emerald green)
    "#e74c3c",  # 2: Boundary (Coral red)
])


def colorize_3class_mask(mask: np.ndarray) -> np.ndarray:
    """Map integer mask {0, 1, 2} to RGB image."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        rgb[mask == c] = COLOR_PALETTE[c]
    return rgb


def find_best_zoom_window(target_mask: np.ndarray, window_size: int = 64) -> Tuple[int, int, int, int]:
    """
    Find the optimal (top, left, bottom, right) bounding box of size window_size x window_size
    maximizing boundary & touching density.
    """
    h, w = target_mask.shape
    if np.sum(target_mask > 0) == 0:
        # Empty patch: return center
        t = (h - window_size) // 2
        l = (w - window_size) // 2
        return t, l, t + window_size, l + window_size

    # Weight boundary pixels heavily (weight 3) and interior (weight 1)
    weight_map = (target_mask == 2).astype(np.float32) * 3.0 + (target_mask == 1).astype(np.float32) * 1.0
    kernel = np.ones((window_size, window_size), dtype=np.float32)
    density = cv2.filter2D(weight_map, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    # Restrict search within valid interior
    half = window_size // 2
    valid_density = density[half:h - half, half:w - half]
    if valid_density.size == 0:
        return 0, 0, window_size, window_size

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(valid_density)
    center_x = max_loc[0] + half
    center_y = max_loc[1] + half

    top = max(0, min(h - window_size, center_y - half))
    left = max(0, min(w - window_size, center_x - half))
    return top, left, top + window_size, left + window_size


def render_4panel_visualization(
    img_rgb: np.ndarray,
    bin_mask: np.ndarray,
    target_mask: np.ndarray,
    patch_info: Dict[str, Any],
    output_path: str,
    zoom_size: int = 64
):
    """
    Render 4-panel visual comparison:
    1. Original aerial RGB
    2. Binary building mask
    3. 3-class target mask
    4. Zoomed view of dense/touching buildings with boundary details
    """
    h, w = target_mask.shape
    color_target = colorize_3class_mask(target_mask)

    # Find zoom box
    zt, zl, zb, zr = find_best_zoom_window(target_mask, window_size=zoom_size)

    # Zoom crops
    zoom_rgb = img_rgb[zt:zb, zl:zr]
    zoom_target = color_target[zt:zb, zl:zr]

    # Create blended zoomed overlay (50% RGB + 50% 3-class target)
    zoom_blend = cv2.addWeighted(zoom_rgb, 0.45, zoom_target, 0.55, 0)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), dpi=200)
    fig.patch.set_facecolor("#0f1117")

    # Titles and subtitles styling
    category_title = patch_info.get("category", "building").replace("_", " ").upper()
    patch_name = patch_info.get("patch_name", "patch")
    split = patch_info.get("split", "train").upper()
    cov = patch_info.get("building_coverage_pct", 0.0)

    # Panel 1: Original RGB Image
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"1. Aerial Image ({split})\n{patch_name}", color="white", fontsize=11, fontweight="bold", pad=8)
    axes[0].axis("off")
    # Draw yellow bounding box indicating zoom region
    rect = patches.Rectangle((zl, zt), zoom_size, zoom_size, linewidth=2, edgecolor="#f1c40f", facecolor="none", linestyle="--")
    axes[0].add_patch(rect)

    # Panel 2: Binary Building Mask
    axes[1].imshow(bin_mask, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"2. Binary Mask (Baseline)\nCoverage: {cov:.1f}%", color="white", fontsize=11, fontweight="bold", pad=8)
    axes[1].axis("off")

    # Panel 3: 3-Class Target Mask
    axes[2].imshow(color_target)
    n_int = np.sum(target_mask == 1)
    n_bnd = np.sum(target_mask == 2)
    axes[2].set_title(f"3. 3-Class Target Mask\nInterior (Green): {n_int}px | Boundary (Red): {n_bnd}px", color="white", fontsize=11, fontweight="bold", pad=8)
    axes[2].axis("off")

    # Panel 4: Zoomed View (High Resolution)
    # Side-by-side zoom in panel 4: split into zoomed binary vs zoomed 3-class overlay
    zoom_bin_rgb = cv2.cvtColor(bin_mask[zt:zb, zl:zr], cv2.COLOR_GRAY2RGB)
    # Combine zoomed binary and zoomed 3-class side-by-side with a separator
    sep = np.ones((zoom_size, 3, 3), dtype=np.uint8) * 255
    zoom_composite = np.concatenate([zoom_rgb, sep, zoom_bin_rgb, sep, zoom_blend], axis=1)

    axes[3].imshow(zoom_composite)
    axes[3].set_title(f"4. Zoom Detail [{zl}:{zr}, {zt}:{zb}]\nAerial  |  Binary (Merged)  |  3-Class (Separated)", color="#f1c40f", fontsize=11, fontweight="bold", pad=8)
    axes[3].axis("off")

    # Legend at bottom
    legend_elements = [
        patches.Patch(facecolor="#1e2028", edgecolor="#555555", label="Class 0: Background"),
        patches.Patch(facecolor="#2ecc71", edgecolor="#27ae60", label="Class 1: Building Interior"),
        patches.Patch(facecolor="#e74c3c", edgecolor="#c0392b", label="Class 2: Building Boundary (1-2px)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, framealpha=0.2,
               facecolor="#1a1c24", edgecolor="#333333", fontsize=10, labelcolor="white", bbox_to_anchor=(0.5, -0.05))

    plt.suptitle(f"CATEGORY: {category_title} — SpaceNet 7 AOI Cadastral Extraction",
                 color="#00ffff", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def create_master_overview_grid(selected_patches: List[Dict[str, Any]], output_path: str):
    """
    Create a grand master grid displaying all representative categories.
    """
    n = len(selected_patches)
    fig, axes = plt.subplots(n, 4, figsize=(18, 3.8 * n), dpi=150)
    fig.patch.set_facecolor("#0b0d13")

    for idx, pinfo in enumerate(selected_patches):
        img_rgb = np.array(Image.open(pinfo["img_path"]).convert("RGB"))
        bin_mask = np.array(Image.open(pinfo["bin_path"]).convert("L"))
        target_mask = np.array(Image.open(pinfo["target_path"]).convert("L"))
        color_target = colorize_3class_mask(target_mask)

        # Zoom window
        zt, zl, zb, zr = find_best_zoom_window(target_mask, window_size=64)
        z_rgb = img_rgb[zt:zb, zl:zr]
        z_target = color_target[zt:zb, zl:zr]
        z_blend = cv2.addWeighted(z_rgb, 0.45, z_target, 0.55, 0)

        cat_label = pinfo.get("category", "").replace("_", " ").upper()
        pname = pinfo.get("patch_name", "")

        # Col 0: Aerial
        axes[idx, 0].imshow(img_rgb)
        rect = patches.Rectangle((zl, zt), 64, 64, linewidth=1.5, edgecolor="#f1c40f", facecolor="none", linestyle="--")
        axes[idx, 0].add_patch(rect)
        axes[idx, 0].set_ylabel(f"[{cat_label}]\n{pname}", color="#00ffff", fontsize=9, fontweight="bold")
        if idx == 0:
            axes[idx, 0].set_title("1. Aerial RGB Image", color="white", fontsize=11, fontweight="bold")
        axes[idx, 0].set_xticks([])
        axes[idx, 0].set_yticks([])

        # Col 1: Binary
        axes[idx, 1].imshow(bin_mask, cmap="gray")
        if idx == 0:
            axes[idx, 1].set_title("2. Binary Mask (Merged)", color="white", fontsize=11, fontweight="bold")
        axes[idx, 1].axis("off")

        # Col 2: 3-Class Target
        axes[idx, 2].imshow(color_target)
        if idx == 0:
            axes[idx, 2].set_title("3. 3-Class Target (Interior/Boundary)", color="white", fontsize=11, fontweight="bold")
        axes[idx, 2].axis("off")

        # Col 3: Zoomed Detail
        axes[idx, 3].imshow(z_blend)
        if idx == 0:
            axes[idx, 3].set_title("4. Zoom Detail (Boundary Split)", color="#f1c40f", fontsize=11, fontweight="bold")
        axes[idx, 3].axis("off")

    plt.suptitle("PHASE 5: 3-CLASS TARGET VALIDATION — REPRESENTATIVE CATEGORY OVERVIEW",
                 color="white", fontsize=15, fontweight="bold", y=1.002)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Master overview grid saved to {output_path}")


def run_visual_validation(
    patch_records_path: str = "results/experiments/3class_targets/patch_records.json",
    output_dir: str = "results/experiments/3class_targets/visualizations"
):
    """
    Select 14 representative patches across all categories and render all figures.
    """
    with open(patch_records_path, "r", encoding="utf-8") as f:
        records = f.load(f) if hasattr(f, "load") else json.load(f)

    # Detailed categorizer
    analyzed = []
    for r in records:
        target = np.array(Image.open(r["target_path"]).convert("L"))
        bin_m = (target > 0).astype(np.uint8)
        interior_m = (target == 1).astype(np.uint8)

        if r["is_empty"]:
            analyzed.append({
                **r, "num_buildings": 0, "max_area": 0, "mean_area": 0,
                "irregularity": 0, "touching_delta": 0, "coord": f"r{r['top']:04d}_c{r['left']:04d}"
            })
            continue

        num_bin, _, stats_bin, _ = cv2.connectedComponentsWithStats(bin_m)
        num_int, _, _, _ = cv2.connectedComponentsWithStats(interior_m)

        areas = stats_bin[1:, cv2.CC_STAT_AREA] if num_bin > 1 else [0]
        max_area = int(np.max(areas))
        mean_area = float(np.mean(areas))
        touching_delta = (num_int - 1) - (num_bin - 1)

        contours, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        irregularity = 0.0
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a > 80:
                p = cv2.arcLength(cnt, True)
                ratio = p / (np.sqrt(a) + 1e-5)
                if ratio > irregularity:
                    irregularity = ratio

        analyzed.append({
            **r,
            "num_buildings": num_bin - 1,
            "max_area": max_area,
            "mean_area": mean_area,
            "irregularity": irregularity,
            "touching_delta": touching_delta,
            "coord": f"r{r['top']:04d}_c{r['left']:04d}"
        })

    selected = []
    used_patches = set()

    def pick(cands, cat, key_fn, reverse=True, n=2):
        s_cands = sorted(cands, key=key_fn, reverse=reverse)
        count = 0
        for c in s_cands:
            if c["patch_name"] not in used_patches:
                selected.append({"category": cat, **c})
                used_patches.add(c["patch_name"])
                count += 1
                if count >= n:
                    break

    # 1. empty_background
    pick([p for p in analyzed if p["is_empty"]], "empty_background", lambda x: x["patch_name"], reverse=False, n=2)

    # 2. isolated_buildings (low coverage, 1-6 discrete buildings, not touching)
    pick([p for p in analyzed if 0.01 <= p["building_coverage_pct"] <= 1.0 and not p["is_touching"]],
         "isolated_buildings", lambda x: x["building_coverage_pct"], reverse=False, n=2)

    # 3. dense_buildings (high coverage, > 14%)
    pick([p for p in analyzed if p["building_coverage_pct"] > 14.0],
         "dense_buildings", lambda x: x["building_coverage_pct"], reverse=True, n=2)

    # 4. touching_adjacent (highest touching delta, fused instances)
    pick([p for p in analyzed if p["touching_delta"] >= 15],
         "touching_adjacent", lambda x: x["touching_delta"], reverse=True, n=2)

    # 5. small_buildings (mean area < 90 px, >= 15 buildings)
    pick([p for p in analyzed if p["mean_area"] < 90 and p["num_buildings"] >= 15],
         "small_buildings", lambda x: x["mean_area"], reverse=False, n=2)

    # 6. large_buildings (max area >= 280 px)
    pick([p for p in analyzed if p["max_area"] >= 280],
         "large_buildings", lambda x: x["max_area"], reverse=True, n=2)

    # 7. irregular_buildings (high perimeter irregularity)
    pick([p for p in analyzed if p["irregularity"] > 5.0],
         "irregular_buildings", lambda x: x["irregularity"], reverse=True, n=2)

    print(f"Selected {len(selected)} representative patches for visual validation across 7 categories.")

    # Render each 4-panel figure
    for idx, pinfo in enumerate(selected, 1):
        cat = pinfo["category"]
        pname = pinfo["patch_name"]
        out_fig = os.path.join(output_dir, f"{idx:02d}_{cat}_{pname}.png")

        img_rgb = np.array(Image.open(pinfo["img_path"]).convert("RGB"))
        bin_mask = np.array(Image.open(pinfo["bin_path"]).convert("L"))
        target_mask = np.array(Image.open(pinfo["target_path"]).convert("L"))

        render_4panel_visualization(
            img_rgb=img_rgb,
            bin_mask=bin_mask,
            target_mask=target_mask,
            patch_info=pinfo,
            output_path=out_fig
        )
        print(f"  [{idx:02d}/{len(selected)}] Rendered {cat} -> {out_fig}")

    # Render grand master grid
    master_path = os.path.join(output_dir, "00_master_3class_overview.png")
    create_master_overview_grid(selected, master_path)

    # Save metadata index of selected patches
    sel_meta_path = os.path.join(os.path.dirname(output_dir), "selected_representative_patches.json")
    with open(sel_meta_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)
    print(f"Visual validation complete. All figures saved in {output_dir}")


if __name__ == "__main__":
    run_visual_validation()
