"""
Rasterize SpaceNet 7 Building Footprints into Binary Masks.

This script extracts RGB channels from a SpaceNet 7 GeoTIFF image,
rasterizes building footprint polygons from labels_match_pix GeoJSON,
and generates aligned (RGB image, binary mask, overlay) for verification.
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import cv2
from PIL import Image
import rasterio


def rasterize_single_pair(
    image_path: str,
    geojson_path: str,
    output_dir: str,
    save_overlay: bool = True
):
    """
    Rasterize building footprints from pixel-space GeoJSON onto a binary mask
    matching the input GeoTIFF image dimensions.
    """
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # 1. Load GeoTIFF imagery using rasterio
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    with rasterio.open(image_path) as src:
        h, w = src.height, src.width
        num_bands = src.count
        # Read bands 1, 2, 3 (RGB)
        r = src.read(1)
        g = src.read(2)
        b = src.read(3)
        rgb = np.stack([r, g, b], axis=-1)  # shape (H, W, 3), uint8

        # Read band 4 (Alpha / Valid mask) if present
        alpha_stats = {}
        if num_bands >= 4:
            alpha = src.read(4)
            alpha_stats = {
                "band_4_present": True,
                "band_4_min": int(alpha.min()),
                "band_4_max": int(alpha.max()),
                "band_4_mean": float(alpha.mean()),
                "valid_pixels_pct": float(np.mean(alpha > 0) * 100)
            }
        else:
            alpha_stats = {"band_4_present": False}

    # 2. Load GeoJSON building footprints
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")

    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    features = geojson_data.get("features", [])
    num_features = len(features)

    # 3. Create binary mask (0 = background, 255 = building)
    mask = np.zeros((h, w), dtype=np.uint8)

    rasterized_count = 0
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
            # Exterior ring
            exterior = np.array(poly[0], dtype=np.float32)
            exterior_pts = np.round(exterior).astype(np.int32)
            cv2.fillPoly(mask, [exterior_pts], color=255)
            rasterized_count += 1

            # Interior rings (holes) if any
            for interior in poly[1:]:
                interior_pts = np.round(np.array(interior, dtype=np.float32)).astype(np.int32)
                cv2.fillPoly(mask, [interior_pts], color=0)

    # Calculate mask statistics
    building_pixels = int(np.sum(mask > 0))
    total_pixels = h * w
    building_coverage_pct = float(building_pixels / total_pixels * 100)

    # 4. Save Outputs
    # Save RGB PNG
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    rgb_save_path = os.path.join(output_dir, f"{base_name}_rgb.png")
    mask_save_path = os.path.join(output_dir, f"{base_name}_mask.png")
    overlay_save_path = os.path.join(output_dir, f"{base_name}_overlay.png")

    Image.fromarray(rgb).save(rgb_save_path)
    Image.fromarray(mask).save(mask_save_path)

    # 5. Create visually striking overlay (RGB + transparent red/green mask + contours)
    if save_overlay:
        overlay = rgb.copy()
        # Highlight buildings in semi-transparent red
        color_mask = np.zeros_like(rgb)
        color_mask[mask > 0] = [255, 60, 60]  # Red for building footprints
        overlay = cv2.addWeighted(overlay, 0.75, color_mask, 0.25, 0)

        # Draw green building contours for sharp cadastral boundaries
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (50, 255, 120), 1, lineType=cv2.LINE_AA)

        Image.fromarray(overlay).save(overlay_save_path)

    elapsed_time = time.time() - t0

    stats = {
        "status": "success",
        "image_file": os.path.basename(image_path),
        "geojson_file": os.path.basename(geojson_path),
        "image_dimensions": [w, h, num_bands],
        "rgb_saved": rgb_save_path,
        "mask_saved": mask_save_path,
        "overlay_saved": overlay_save_path if save_overlay else None,
        "features_in_geojson": num_features,
        "polygons_rasterized": rasterized_count,
        "building_pixels": building_pixels,
        "total_pixels": total_pixels,
        "building_coverage_pct": round(building_coverage_pct, 3),
        "alpha_band_stats": alpha_stats,
        "elapsed_seconds": round(elapsed_time, 3)
    }

    stats_save_path = os.path.join(output_dir, f"{base_name}_stats.json")
    with open(stats_save_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Rasterize SpaceNet 7 Sample Pair")
    parser.add_argument(
        "--dataset-root",
        default=r"sample/L15-0506E-1204N_2027_3374_13",
        help="Path to SpaceNet 7 AOI directory"
    )
    parser.add_argument(
        "--sample-month",
        default="2018_01",
        help="Month identifier (e.g. 2018_01)"
    )
    parser.add_argument(
        "--output-dir",
        default=r"results/dataset_inspection/sample_verification",
        help="Directory to save generated mask and verification images"
    )

    args = parser.parse_args()

    image_dir = os.path.join(args.dataset_root, "images")
    label_dir = os.path.join(args.dataset_root, "labels_match_pix")

    # Find matching image and label
    images = [f for f in os.listdir(image_dir) if f.endswith(".tif") and args.sample_month in f]
    if not images:
        print(f"Error: No image found matching month {args.sample_month} in {image_dir}")
        sys.exit(1)

    image_fname = images[0]
    image_path = os.path.join(image_dir, image_fname)

    label_fname = image_fname.replace(".tif", "_Buildings.geojson")
    label_path = os.path.join(label_dir, label_fname)

    if not os.path.exists(label_path):
        print(f"Error: Matching GeoJSON not found: {label_path}")
        sys.exit(1)

    print(f"==================================================")
    print(f"SpaceNet 7 Sample Rasterization Verification")
    print(f"==================================================")
    print(f"Image:   {image_path}")
    print(f"Label:   {label_path}")
    print(f"Output:  {args.output_dir}")
    print(f"--------------------------------------------------")

    stats = rasterize_single_pair(image_path, label_path, args.output_dir)

    print(f"Successfully rasterized!")
    print(f"  - Image Dimensions:      {stats['image_dimensions'][0]}x{stats['image_dimensions'][1]} ({stats['image_dimensions'][2]} bands)")
    print(f"  - Features in GeoJSON:   {stats['features_in_geojson']}")
    print(f"  - Polygons Rasterized:   {stats['polygons_rasterized']}")
    print(f"  - Building Coverage:     {stats['building_coverage_pct']}% ({stats['building_pixels']} px)")
    print(f"  - Elapsed Time:          {stats['elapsed_seconds']}s")
    print(f"  - RGB Image Saved:       {stats['rgb_saved']}")
    print(f"  - Binary Mask Saved:     {stats['mask_saved']}")
    print(f"  - Cadastral Overlay:     {stats['overlay_saved']}")
    print(f"==================================================")


if __name__ == "__main__":
    main()
