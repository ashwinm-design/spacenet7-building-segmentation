"""
SpaceNet 7 Full Dataset Preparation & Patch-Tiling Pipeline.

This script:
1. Discovers and sorts all 24 monthly scenes for AOI L15-0506E-1204N_2027_3374_13.
2. Rasterizes building footprints from labels_match_pix into 1024x1024 binary masks.
3. Extracts RGB imagery from GeoTIFFs (bands 1, 2, 3).
4. Tiles each 1024x1024 scene into non-overlapping 256x256 patches (16 per scene).
5. Splits dataset temporally:
   - Train: 16 months (2018-01 through 2019-04) -> 256 patches
   - Val:    4 months (2019-05 through 2019-08) -> 64 patches
   - Test:   4 months (2019-09 through 2019-12) -> 64 patches
6. Writes split files compatible with PyTorch DataLoader (train.txt, val.txt, test.txt).
"""

import os
import sys
import re
import json
import time
import argparse
import numpy as np
import cv2
from PIL import Image
import rasterio


def parse_month_from_filename(filename: str) -> str:
    """Extract YYYY_MM from SpaceNet 7 filename."""
    match = re.search(r"global_monthly_(\d{4}_\d{2})", filename)
    if match:
        return match.group(1)
    return filename


def rasterize_single_scene(image_path: str, geojson_path: str):
    """
    Load RGB from GeoTIFF and rasterize building polygons from GeoJSON.
    Returns: (rgb_array, mask_array, num_polygons, building_pixels)
    """
    with rasterio.open(image_path) as src:
        h, w = src.height, src.width
        r = src.read(1)
        g = src.read(2)
        b = src.read(3)
        rgb = np.stack([r, g, b], axis=-1)

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mask = np.zeros((h, w), dtype=np.uint8)
    features = data.get("features", [])
    poly_count = 0

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
            if not poly:
                continue
            exterior = np.round(np.array(poly[0], dtype=np.float32)).astype(np.int32)
            cv2.fillPoly(mask, [exterior], color=255)
            poly_count += 1

            for interior in poly[1:]:
                interior_pts = np.round(np.array(interior, dtype=np.float32)).astype(np.int32)
                cv2.fillPoly(mask, [interior_pts], color=0)

    building_pixels = int(np.sum(mask > 0))
    return rgb, mask, poly_count, building_pixels


def tile_scene(
    rgb: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 256,
    stride: int = 256
):
    """
    Slice an (H, W, 3) image and (H, W) mask into patches of (patch_size, patch_size).
    Returns list of dicts with patch arrays, coordinates, and statistics.
    """
    h, w = rgb.shape[:2]
    patches = []

    for top in range(0, h - patch_size + 1, stride):
        for left in range(0, w - patch_size + 1, stride):
            bottom = top + patch_size
            right = left + patch_size

            patch_rgb = rgb[top:bottom, left:right]
            patch_mask = mask[top:bottom, left:right]

            b_pixels = int(np.sum(patch_mask > 0))
            total_p = patch_size * patch_size
            cov_pct = float(b_pixels / total_p * 100)

            patches.append({
                "top": top,
                "left": left,
                "rgb": patch_rgb,
                "mask": patch_mask,
                "building_pixels": b_pixels,
                "coverage_pct": round(cov_pct, 2)
            })

    return patches


def prepare_spacenet7_dataset(
    dataset_root: str,
    output_dir: str,
    patch_size: int = 256,
    stride: int = 256,
    train_months: int = 16,
    val_months: int = 4,
    test_months: int = 4
):
    """
    Run the end-to-end dataset preparation and tiling pipeline.
    """
    t_start = time.time()
    images_dir = os.path.join(dataset_root, "images")
    labels_dir = os.path.join(dataset_root, "labels_match_pix")

    # Output directory layout
    full_img_dir = os.path.join(output_dir, "full_scenes", "images")
    full_mask_dir = os.path.join(output_dir, "full_scenes", "masks")
    patches_img_dir = os.path.join(output_dir, "patches", "images")
    patches_mask_dir = os.path.join(output_dir, "patches", "masks")
    splits_dir = os.path.join(output_dir, "splits")

    for d in [full_img_dir, full_mask_dir, splits_dir]:
        os.makedirs(d, exist_ok=True)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(patches_img_dir, split), exist_ok=True)
        os.makedirs(os.path.join(patches_mask_dir, split), exist_ok=True)

    # Collect and sort all monthly image files chronologically
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".tif")])
    num_scenes = len(image_files)

    if num_scenes == 0:
        raise FileNotFoundError(f"No GeoTIFF images found in {images_dir}")

    print(f"Discovered {num_scenes} monthly scenes in {dataset_root}")

    # Determine temporal splits
    # e.g., 24 months total -> 16 train, 4 val, 4 test
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
    print(f"  - Train: {len(train_files)} months ({parse_month_from_filename(train_files[0])} to {parse_month_from_filename(train_files[-1])})")
    print(f"  - Val:   {len(val_files)} months ({parse_month_from_filename(val_files[0])} to {parse_month_from_filename(val_files[-1])})")
    print(f"  - Test:  {len(test_files)} months ({parse_month_from_filename(test_files[0])} to {parse_month_from_filename(test_files[-1])})")

    split_records = {"train": [], "val": [], "test": []}
    scene_summaries = []
    total_patches = 0
    total_buildings = 0

    for idx, img_fname in enumerate(image_files, 1):
        month_str = parse_month_from_filename(img_fname)
        split = split_map[img_fname]

        img_path = os.path.join(images_dir, img_fname)
        geojson_fname = img_fname.replace(".tif", "_Buildings.geojson")
        geojson_path = os.path.join(labels_dir, geojson_fname)

        if not os.path.exists(geojson_path):
            print(f"Warning: GeoJSON missing for {img_fname}, skipping...")
            continue

        # 1. Rasterize scene
        rgb, mask, poly_count, building_px = rasterize_single_scene(img_path, geojson_path)
        total_buildings += poly_count

        # Save full scene PNGs
        base_name = f"{month_str}_scene"
        full_img_path = os.path.join(full_img_dir, f"{base_name}.png")
        full_mask_path = os.path.join(full_mask_dir, f"{base_name}_mask.png")

        Image.fromarray(rgb).save(full_img_path)
        Image.fromarray(mask).save(full_mask_path)

        # 2. Slice into patches
        patches = tile_scene(rgb, mask, patch_size=patch_size, stride=stride)

        for p_idx, p in enumerate(patches):
            patch_name = f"{month_str}_r{p['top']:04d}_c{p['left']:04d}"
            dst_patch_img = os.path.join(patches_img_dir, split, f"{patch_name}.png")
            dst_patch_mask = os.path.join(patches_mask_dir, split, f"{patch_name}_mask.png")

            Image.fromarray(p["rgb"]).save(dst_patch_img)
            Image.fromarray(p["mask"]).save(dst_patch_mask)

            # Store absolute / normalized relative paths for splits
            split_records[split].append(f"{dst_patch_img},{dst_patch_mask}\n")
            total_patches += 1

        scene_summaries.append({
            "month": month_str,
            "split": split,
            "polygons": poly_count,
            "building_coverage_pct": round(building_px / (rgb.shape[0] * rgb.shape[1]) * 100, 2),
            "num_patches": len(patches)
        })

        if idx % 6 == 0 or idx == num_scenes:
            print(f"  Processed {idx}/{num_scenes} scenes ({month_str}) — {poly_count} buildings rasterized")

    # 3. Write split text files
    for split_name, lines in split_records.items():
        split_file = os.path.join(splits_dir, f"{split_name}.txt")
        with open(split_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"Wrote {len(lines)} patch pairs to {split_file}")

    elapsed = time.time() - t_start

    # Summary metadata JSON
    metadata = {
        "dataset_name": "SpaceNet 7 Cadastral Patch Dataset",
        "aoi": "L15-0506E-1204N_2027_3374_13",
        "patch_size": [patch_size, patch_size],
        "stride": stride,
        "total_scenes": num_scenes,
        "total_patches": total_patches,
        "patches_per_split": {k: len(v) for k, v in split_records.items()},
        "total_building_footprints_rasterized": total_buildings,
        "elapsed_seconds": round(elapsed, 2),
        "scene_summaries": scene_summaries
    }

    summary_path = os.path.join(output_dir, "spacenet7_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n==================================================")
    print(f"SpaceNet 7 Preparation & Tiling Complete!")
    print(f"  - Total Scenes:       {num_scenes}")
    print(f"  - Total Patches:      {total_patches} ({patch_size}x{patch_size})")
    print(f"  - Train Patches:      {len(split_records['train'])}")
    print(f"  - Val Patches:        {len(split_records['val'])}")
    print(f"  - Test Patches:       {len(split_records['test'])}")
    print(f"  - Total Elapsed:      {round(elapsed, 2)}s")
    print(f"  - Output Directory:   {output_dir}")
    print(f"==================================================")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Batch Rasterize & Tile SpaceNet 7 Dataset")
    parser.add_argument(
        "--dataset-root",
        default=r"sample/L15-0506E-1204N_2027_3374_13",
        help="Root directory of SpaceNet 7 sample AOI"
    )
    parser.add_argument(
        "--output-dir",
        default=r"data/processed_spacenet7",
        help="Output directory for processed scenes, patches, and splits"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch tile size (default 256)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Patch tile stride (default 256 for non-overlapping)"
    )
    args = parser.parse_args()

    prepare_spacenet7_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride
    )


if __name__ == "__main__":
    main()
