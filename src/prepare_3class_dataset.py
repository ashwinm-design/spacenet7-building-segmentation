"""
SpaceNet 7 Phase 5: 3-Class Target Dataset Preparation Pipeline.

Generates:
  Class 0: Background
  Class 1: Building Interior
  Class 2: Building Boundary

The boundary is derived from individual ground-truth building polygons:
- For each polygon, rasterizes the full footprint (including holes).
- Computes an inner boundary band (1-2 pixels wide) via morphological erosion (3x3 structuring element).
- At contact interfaces where adjacent buildings touch or are closely adjacent, boundary takes priority,
  forming an explicit 1-2 pixel separating barrier between interior regions.
- Boundary labels are strictly constrained within the building footprints, with 0 spillover into true background.

Maintains identical 16 train / 4 val / 4 test temporal split (256 train, 64 val, 64 test patches).
"""

import os
import sys
import re
import json
import time
import shutil
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import cv2
from PIL import Image
import rasterio

# Ensure standard output uses UTF-8 to prevent Windows CP1252 charmap crashes
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_month_from_filename(filename: str) -> str:
    """Extract YYYY_MM from SpaceNet 7 filename."""
    match = re.search(r"global_monthly_(\d{4}_\d{2})", filename)
    if match:
        return match.group(1)
    return filename


def rasterize_scene_3class(image_path: str, geojson_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load RGB from GeoTIFF and rasterize building polygons into:
      1. Binary building mask (0: bg, 255: building)
      2. 3-class target mask (0: bg, 1: interior, 2: boundary)
    
    Returns:
      rgb (H, W, 3) uint8
      binary_mask (H, W) uint8 {0, 255}
      target_mask (H, W) uint8 {0, 1, 2}
      stats dict
    """
    with rasterio.open(image_path) as src:
        h, w = src.height, src.width
        r = src.read(1)
        g = src.read(2)
        b = src.read(3)
        rgb = np.stack([r, g, b], axis=-1)

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    binary_mask = np.zeros((h, w), dtype=np.uint8)
    total_interior = np.zeros((h, w), dtype=bool)
    total_boundary = np.zeros((h, w), dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    features = data.get("features", [])
    poly_count = 0
    tiny_poly_count = 0

    for feat in features:
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])

        if gtype == "Polygon":
            polys = [coords]
        elif gtype == "MultiPolygon":
            polys = coords
        else:
            continue

        for poly in polys:
            if not poly or len(poly[0]) < 3:
                continue

            exterior = np.round(np.array(poly[0], dtype=np.float32)).astype(np.int32)
            xmin, ymin = np.min(exterior, axis=0)
            xmax, ymax = np.max(exterior, axis=0)

            pad = 2
            x0, y0 = max(0, int(xmin - pad)), max(0, int(ymin - pad))
            x1, y1 = min(w, int(xmax + pad + 1)), min(h, int(ymax + pad + 1))
            lh, lw = y1 - y0, x1 - x0

            if lh <= 0 or lw <= 0:
                continue

            local_poly = exterior - np.array([x0, y0])
            local_mask = np.zeros((lh, lw), dtype=np.uint8)
            cv2.fillPoly(local_mask, [local_poly], 255)

            # Rasterize interior holes (if any)
            for hole in poly[1:]:
                if len(hole) >= 3:
                    inter_pts = np.round(np.array(hole, dtype=np.float32)).astype(np.int32) - np.array([x0, y0])
                    cv2.fillPoly(local_mask, [inter_pts], 0)

            poly_pixels = int(np.sum(local_mask > 0))
            if poly_pixels == 0:
                continue

            poly_count += 1
            # Also fill full binary mask
            binary_mask[y0:y1, x0:x1] |= local_mask

            # 1-pixel inner erosion
            eroded = cv2.erode(local_mask, kernel, iterations=1)

            if np.sum(eroded > 0) == 0:
                # Tiny or narrow polygon fallback: keep centroid/deepest pixel as interior
                tiny_poly_count += 1
                dist = cv2.distanceTransform(local_mask, cv2.DIST_L2, 3)
                max_d = np.max(dist)
                if max_d > 0:
                    candidates = np.argwhere((dist == max_d) & (local_mask > 0))
                    # Pick median candidate to maintain central interior seed
                    med_idx = len(candidates) // 2
                    eroded[candidates[med_idx][0], candidates[med_idx][1]] = 255
                else:
                    coords_nz = np.argwhere(local_mask > 0)
                    cy, cx = np.round(np.mean(coords_nz, axis=0)).astype(int)
                    eroded[cy, cx] = 255

            local_boundary = (local_mask > 0) & (eroded == 0)
            local_interior = (eroded > 0)

            total_interior[y0:y1, x0:x1] |= local_interior
            total_boundary[y0:y1, x0:x1] |= local_boundary

    # Build 3-class target: Background = 0, Interior = 1, Boundary = 2 (priority)
    target_mask = np.zeros((h, w), dtype=np.uint8)
    target_mask[total_interior] = 1
    target_mask[total_boundary] = 2

    # Verification: target_mask > 0 must exactly equal binary_mask > 0
    mismatch = np.sum((target_mask > 0) != (binary_mask > 0))
    if mismatch > 0:
        print(f"Warning: {mismatch} pixels mismatch between binary and 3-class mask in {image_path}")

    stats = {
        "num_polygons": poly_count,
        "tiny_polygons": tiny_poly_count,
        "building_pixels": int(np.sum(binary_mask > 0)),
        "bg_pixels": int(np.sum(target_mask == 0)),
        "interior_pixels": int(np.sum(target_mask == 1)),
        "boundary_pixels": int(np.sum(target_mask == 2)),
    }

    return rgb, binary_mask, target_mask, stats


def detect_touching_buildings(patch_binary: np.ndarray, patch_target: np.ndarray) -> bool:
    """
    Detect if a patch contains touching/adjacent buildings that are merged in binary
    but separated in the 3-class interior representation.
    """
    if np.sum(patch_binary > 0) == 0:
        return False

    num_binary_comps, _ = cv2.connectedComponents((patch_binary > 0).astype(np.uint8))
    num_interior_comps, _ = cv2.connectedComponents((patch_target == 1).astype(np.uint8))

    # If interior has more disconnected components than the binary mask,
    # two or more instances were fused in the binary mask!
    return (num_interior_comps - 1) > (num_binary_comps - 1)


def generate_3class_dataset(
    dataset_root: str = "sample/L15-0506E-1204N_2027_3374_13",
    output_dir: str = "data/processed_spacenet7_3class",
    stats_output_json: str = "results/experiments/3class_targets/target_statistics.json",
    patch_size: int = 256,
    stride: int = 256,
    train_months: int = 16,
    val_months: int = 4,
    test_months: int = 4
) -> Dict[str, Any]:
    """
    End-to-end dataset generator for 3-class SpaceNet 7 semantic segmentation.
    """
    t0 = time.time()
    images_dir = os.path.join(dataset_root, "images")
    labels_dir = os.path.join(dataset_root, "labels_match_pix")

    # Directory structure requested:
    # data/processed_spacenet7_3class/
    #   images/
    #   masks/
    #   splits/
    # We organize split subfolders inside images/ and masks/
    out_img_dir = os.path.join(output_dir, "images")
    out_mask_dir = os.path.join(output_dir, "masks")
    out_bin_dir = os.path.join(output_dir, "binary_masks")
    out_splits_dir = os.path.join(output_dir, "splits")

    for d in [out_img_dir, out_mask_dir, out_bin_dir, out_splits_dir]:
        os.makedirs(d, exist_ok=True)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(out_img_dir, split), exist_ok=True)
        os.makedirs(os.path.join(out_mask_dir, split), exist_ok=True)
        os.makedirs(os.path.join(out_bin_dir, split), exist_ok=True)

    # Collect and sort all 24 monthly scene images
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".tif")])
    num_scenes = len(image_files)

    if num_scenes == 0:
        raise FileNotFoundError(f"No GeoTIFF scenes found in {images_dir}")

    print(f"Discovered {num_scenes} scenes in {dataset_root}")

    train_files = image_files[:train_months]
    val_files = image_files[train_months:train_months + val_months]
    test_files = image_files[train_months + val_months:train_months + val_months + test_months]

    split_map = {}
    for f in train_files:
        split_map[f] = "train"
    for f in val_files:
        split_map[f] = "val"
    for f in test_files:
        split_map[f] = "test"

    print(f"Temporal Split Strategy:")
    print(f"  - Train: {len(train_files)} scenes -> {len(train_files)*16} patches")
    print(f"  - Val:   {len(val_files)} scenes -> {len(val_files)*16} patches")
    print(f"  - Test:  {len(test_files)} scenes -> {len(test_files)*16} patches")

    split_records = {"train": [], "val": [], "test": []}
    patch_records = []
    
    total_pixels_global = 0
    bg_pixels_global = 0
    interior_pixels_global = 0
    boundary_pixels_global = 0
    empty_patches_count = 0
    touching_patches_count = 0

    split_stats = {
        "train": {"bg_pixels": 0, "interior_pixels": 0, "boundary_pixels": 0, "total_pixels": 0, "empty_patches": 0, "touching_patches": 0, "patches": 0},
        "val":   {"bg_pixels": 0, "interior_pixels": 0, "boundary_pixels": 0, "total_pixels": 0, "empty_patches": 0, "touching_patches": 0, "patches": 0},
        "test":  {"bg_pixels": 0, "interior_pixels": 0, "boundary_pixels": 0, "total_pixels": 0, "empty_patches": 0, "touching_patches": 0, "patches": 0},
    }

    # Process all scenes
    for s_idx, img_fname in enumerate(image_files, 1):
        month_str = parse_month_from_filename(img_fname)
        split = split_map[img_fname]

        img_path = os.path.join(images_dir, img_fname)
        geojson_fname = img_fname.replace(".tif", "_Buildings.geojson")
        geojson_path = os.path.join(labels_dir, geojson_fname)

        if not os.path.exists(geojson_path):
            print(f"Warning: GeoJSON missing for {img_fname}, skipping...")
            continue

        rgb, bin_mask, target_mask, scene_stats = rasterize_scene_3class(img_path, geojson_path)

        # Slice 1024x1024 scene into 16 non-overlapping 256x256 patches
        h, w = rgb.shape[:2]
        for top in range(0, h - patch_size + 1, stride):
            for left in range(0, w - patch_size + 1, stride):
                bottom = top + patch_size
                right = left + patch_size

                p_rgb = rgb[top:bottom, left:right]
                p_bin = bin_mask[top:bottom, left:right]
                p_target = target_mask[top:bottom, left:right]

                # Sanity check dimensions & values
                assert p_rgb.shape == (256, 256, 3), f"Invalid RGB shape: {p_rgb.shape}"
                assert p_target.shape == (256, 256), f"Invalid target shape: {p_target.shape}"
                assert not np.isnan(p_target).any(), "NaN found in target mask!"
                unique_classes = np.unique(p_target)
                assert all(c in [0, 1, 2] for c in unique_classes), f"Invalid class values: {unique_classes}"

                p_bg = int(np.sum(p_target == 0))
                p_int = int(np.sum(p_target == 1))
                p_bnd = int(np.sum(p_target == 2))
                p_tot = patch_size * patch_size

                is_empty = (p_int == 0 and p_bnd == 0)
                is_touching = detect_touching_buildings(p_bin, p_target)

                if is_empty:
                    empty_patches_count += 1
                    split_stats[split]["empty_patches"] += 1
                if is_touching:
                    touching_patches_count += 1
                    split_stats[split]["touching_patches"] += 1

                total_pixels_global += p_tot
                bg_pixels_global += p_bg
                interior_pixels_global += p_int
                boundary_pixels_global += p_bnd

                split_stats[split]["bg_pixels"] += p_bg
                split_stats[split]["interior_pixels"] += p_int
                split_stats[split]["boundary_pixels"] += p_bnd
                split_stats[split]["total_pixels"] += p_tot
                split_stats[split]["patches"] += 1

                patch_name = f"{month_str}_r{top:04d}_c{left:04d}"
                dst_img = os.path.join(out_img_dir, split, f"{patch_name}.png")
                dst_target = os.path.join(out_mask_dir, split, f"{patch_name}_mask.png")
                dst_bin = os.path.join(out_bin_dir, split, f"{patch_name}_mask.png")

                # Save RGB image
                Image.fromarray(p_rgb).save(dst_img)
                # Save 3-class target mask as uint8 PNG with exact integer classes 0, 1, 2
                Image.fromarray(p_target).save(dst_target)
                # Save binary mask for inspection / reference
                Image.fromarray(p_bin).save(dst_bin)

                # Store relative/normalized paths for splits file
                rel_img = f"data/processed_spacenet7_3class/images/{split}/{patch_name}.png"
                rel_target = f"data/processed_spacenet7_3class/masks/{split}/{patch_name}_mask.png"
                split_records[split].append(f"{rel_img},{rel_target}\n")

                patch_records.append({
                    "patch_name": patch_name,
                    "split": split,
                    "top": top,
                    "left": left,
                    "bg_pixels": p_bg,
                    "interior_pixels": p_int,
                    "boundary_pixels": p_bnd,
                    "is_empty": is_empty,
                    "is_touching": is_touching,
                    "building_coverage_pct": round((p_int + p_bnd) / p_tot * 100, 2),
                    "img_path": dst_img,
                    "target_path": dst_target,
                    "bin_path": dst_bin,
                })

        if s_idx % 6 == 0 or s_idx == num_scenes:
            print(f"  Processed {s_idx}/{num_scenes} scenes ({month_str}) — Cumulative patches: {len(patch_records)}")

    # Write split files
    for split_name, lines in split_records.items():
        split_file = os.path.join(out_splits_dir, f"{split_name}.txt")
        with open(split_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"Wrote {len(lines)} patch pairs to {split_file}")

    elapsed = time.time() - t0

    # Compute global and split percentages
    bg_pct = round(bg_pixels_global / total_pixels_global * 100, 4)
    interior_pct = round(interior_pixels_global / total_pixels_global * 100, 4)
    boundary_pct = round(boundary_pixels_global / total_pixels_global * 100, 4)

    for sp in split_stats:
        tot_sp = split_stats[sp]["total_pixels"]
        split_stats[sp]["bg_pct"] = round(split_stats[sp]["bg_pixels"] / tot_sp * 100, 4)
        split_stats[sp]["interior_pct"] = round(split_stats[sp]["interior_pixels"] / tot_sp * 100, 4)
        split_stats[sp]["boundary_pct"] = round(split_stats[sp]["boundary_pixels"] / tot_sp * 100, 4)
        split_stats[sp]["empty_patches_pct"] = round(split_stats[sp]["empty_patches"] / split_stats[sp]["patches"] * 100, 2)
        split_stats[sp]["touching_patches_pct"] = round(split_stats[sp]["touching_patches"] / split_stats[sp]["patches"] * 100, 2)

    stats_summary = {
        "dataset_name": "SpaceNet 7 3-Class Semantic Segmentation Dataset",
        "aoi": "L15-0506E-1204N_2027_3374_13",
        "patch_size": [patch_size, patch_size],
        "stride": stride,
        "classes": {
            "0": "Background",
            "1": "Building Interior",
            "2": "Building Boundary"
        },
        "boundary_specification": {
            "method": "Per-polygon inner erosion via 3x3 structuring element",
            "boundary_width_isolated_pixels": 1,
            "boundary_width_touching_pixels": 2,
            "boundary_within_footprint_only": True,
            "boundary_priority_over_interior": True
        },
        "total_scenes": num_scenes,
        "total_patches": len(patch_records),
        "split_patch_counts": {
            "train": len(split_records["train"]),
            "val": len(split_records["val"]),
            "test": len(split_records["test"])
        },
        "pixel_counts": {
            "background": bg_pixels_global,
            "interior": interior_pixels_global,
            "boundary": boundary_pixels_global,
            "total": total_pixels_global
        },
        "pixel_percentages": {
            "background": bg_pct,
            "interior": interior_pct,
            "boundary": boundary_pct
        },
        "patch_analysis": {
            "total_empty_patches": empty_patches_count,
            "empty_patches_pct": round(empty_patches_count / len(patch_records) * 100, 2),
            "total_touching_adjacent_patches": touching_patches_count,
            "touching_adjacent_patches_pct": round(touching_patches_count / len(patch_records) * 100, 2)
        },
        "split_breakdown": split_stats,
        "verification": {
            "all_pixels_valid_class": True,
            "no_nan_values": True,
            "dimensions_consistent_256x256": True,
            "split_counts_match_existing": (
                len(split_records["train"]) == 256 and
                len(split_records["val"]) == 64 and
                len(split_records["test"]) == 64
            )
        },
        "elapsed_seconds": round(elapsed, 2)
    }

    # Save target_statistics.json
    os.makedirs(os.path.dirname(stats_output_json), exist_ok=True)
    with open(stats_output_json, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2)
    print(f"Target statistics saved to {stats_output_json}")

    # Also save patch index for visual selection
    patch_index_path = os.path.join(os.path.dirname(stats_output_json), "patch_records.json")
    with open(patch_index_path, "w", encoding="utf-8") as f:
        json.dump(patch_records, f, indent=2)
    print(f"Patch records index saved to {patch_index_path}")

    return stats_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SpaceNet 7 3-Class Target Dataset")
    parser.add_argument("--dataset_root", type=str, default="sample/L15-0506E-1204N_2027_3374_13")
    parser.add_argument("--output_dir", type=str, default="data/processed_spacenet7_3class")
    parser.add_argument("--stats_output", type=str, default="results/experiments/3class_targets/target_statistics.json")
    args = parser.parse_args()

    stats = generate_3class_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        stats_output_json=args.stats_output
    )
    print("\nDataset Preparation Completed Successfully!")
    print(json.dumps(stats, indent=2))
