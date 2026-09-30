#!/usr/bin/env python3
"""
src/run_inference.py

SIH26012 Prototype Inference Entry Point:
End-to-end building-footprint segmentation and vector extraction pipeline.

Usage:
    python src/run_inference.py --input <IMAGE_PATH> --output_dir results/prototype
    python src/run_inference.py --help
"""

import os
import sys
import time
import argparse
import json
import warnings
from typing import Optional

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.prototype.config import InferenceConfig
from src.prototype.geospatial import read_geospatial_metadata, GeospatialMetadata
from src.prototype.preprocessing import load_image_and_metadata, create_tile_grid
from src.prototype.inference import BuildingSegmentor
from src.prototype.polygonization import (
    extract_building_polygons,
    build_footprint_geodataframe,
    save_inference_outputs,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SIH26012: High-Resolution Aerial Building Footprint Inference Backend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input aerial/drone/satellite image (GeoTIFF, PNG, JPEG, etc.).",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="results/prototype",
        help="Root directory where predictions, statistics, and visualizations will be stored.",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="results/experiments/open_cities_adaptation/checkpoints/imagenet_ablation_best_model.pth",
        help="Path to the frozen 3-class model checkpoint.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=256,
        help="Tiling dimension (pixels).",
    )
    parser.add_argument(
        "--tile_stride",
        type=int,
        default=256,
        help="Stride between tile steps (pixels). Equal to tile_size for seamless non-overlapping tiling.",
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=16,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--growth_distance",
        type=float,
        default=1.0,
        help="Euclidean boundary growth distance (pixels) for Class-1 instance expansion.",
    )
    parser.add_argument(
        "--min_area",
        type=float,
        default=15.0,
        help="Minimum building footprint area in pixels to filter out noise.",
    )
    parser.add_argument(
        "--simplify_tol",
        type=float,
        default=0.5,
        help="Polygon vertex simplification tolerance (pixels).",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        default=None,
        help="Optional sub-window region of interest in pixel coordinates: xmin ymin xmax ymax.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device for inference.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    print("=" * 75)
    print("SIH26012: BUILDING-FOOTPRINT INFERENCE BACKEND")
    print("=" * 75)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")

    input_basename = os.path.splitext(os.path.basename(args.input))[0]

    # Initialize configuration
    cfg = InferenceConfig(
        checkpoint_path=args.checkpoint,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
        batch_size=args.batch_size,
        growth_distance=args.growth_distance,
        min_area_pixels=args.min_area,
        simplify_tolerance=args.simplify_tol,
        output_dir=args.output_dir,
        device=args.device,
    )

    print(f"Input file         : {args.input}")
    print(f"Output directory   : {cfg.output_dir}")
    print(f"Model Checkpoint   : {cfg.checkpoint_path}")
    print(f"Boundary Growth    : {cfg.growth_distance:.1f} px (Class-1 interior + boundary expansion)")
    print(f"Tile Configuration : {cfg.tile_size}x{cfg.tile_size} (stride: {cfg.tile_stride})")

    # 1. Load Image and Metadata
    print("\n[Step 1/5] Loading image and reading metadata...")
    rgb_image, valid_mask, geo_meta = load_image_and_metadata(args.input)
    h, w, c = rgb_image.shape

    print(f"  Dimensions       : {w} x {h} (pixels, {c} bands)")
    print(f"  Is GeoTIFF       : {geo_meta.is_geotiff}")
    if geo_meta.is_geotiff:
        print(f"  Source CRS       : {geo_meta.crs}")
        print(f"  Projected CRS    : {geo_meta.is_projected}")
        if geo_meta.resolution:
            print(f"  GSD / Resolution : {geo_meta.resolution[0]:.4f} m/px")
    else:
        print("  Georeference     : None (Ordinary RGB image space)")

    # 2. Handle optional ROI
    if args.roi:
        xmin, ymin, xmax, ymax = args.roi
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)
        print(f"  Applying ROI crop: [{xmin}, {ymin}, {xmax}, {ymax}] -> {xmax-xmin} x {ymax-ymin} px")
        rgb_image = rgb_image[ymin:ymax, xmin:xmax]
        valid_mask = valid_mask[ymin:ymax, xmin:xmax]

        # Adjust GeoTIFF affine transform for the window
        if geo_meta.is_geotiff and geo_meta.transform is not None:
            new_transform = geo_meta.transform * Affine.translation(xmin, ymin)
            geo_meta.transform = new_transform
            geo_meta.width = xmax - xmin
            geo_meta.height = ymax - ymin
        h, w = rgb_image.shape[:2]

    # 3. Create Tile Grid & Load Model
    print("\n[Step 2/5] Initializing segmentor and creating tile grid...")
    tile_grid = create_tile_grid((h, w), cfg.tile_size, cfg.tile_stride)
    print(f"  Total 256x256 tiles to evaluate: {len(tile_grid):,}")

    segmentor = BuildingSegmentor(cfg)
    print(f"  Compute device: {segmentor.device}")

    # 4. Run Model Inference
    print("\n[Step 3/5] Running batched inference and tile stitching...")
    t_inf_start = time.time()
    pred_class, building_prob, _ = segmentor.predict_full_image(rgb_image, valid_mask, tile_grid)
    t_inf_duration = time.time() - t_inf_start
    print(f"  Inference completed in {t_inf_duration:.2f} s ({len(tile_grid)/max(t_inf_duration, 1e-4):.1f} tiles/s)")

    # 5. Boundary-Aware Polygonization & Metric Extraction
    print("\n[Step 4/5] Extracting building instances and polygonizing...")
    t_poly_start = time.time()
    pixel_polys, attributes, _ = extract_building_polygons(pred_class, building_prob, cfg)
    print(f"  Extracted {len(pixel_polys):,} building instances.")

    gdf = build_footprint_geodataframe(pixel_polys, attributes, geo_meta)
    t_poly_duration = time.time() - t_poly_start
    print(f"  Polygonization & metric calculation completed in {t_poly_duration:.2f} s")

    # 6. Save Outputs
    print("\n[Step 5/5] Exporting predictions, statistics, and overlay visualization...")
    saved_files = save_inference_outputs(gdf, rgb_image, pixel_polys, input_basename, geo_meta, cfg)

    total_time = time.time() - t_start
    total_bldgs = len(pixel_polys)
    total_area = float(gdf["area"].sum()) if total_bldgs > 0 else 0.0
    mean_conf = float(gdf["confidence"].mean()) if total_bldgs > 0 else 0.0
    area_unit = gdf["area_unit"].iloc[0] if total_bldgs > 0 else "pixels^2"

    print("\n" + "=" * 75)
    print("INFERENCE SUMMARY & TECHNICAL METRICS")
    print("=" * 75)
    print(f"Input Scene           : {input_basename}")
    print(f"Detected Buildings    : {total_bldgs:,}")
    print(f"Total Footprint Area  : {total_area:,.2f} {area_unit}")
    print(f"Mean Confidence       : {mean_conf:.4f}")
    print(f"CRS Preserved         : {geo_meta.crs if geo_meta.is_geotiff else 'N/A (Pixel Coordinates)'}")
    print(f"Total Processing Time : {total_time:.2f} s (Inference: {t_inf_duration:.2f} s, Polygonization: {t_poly_duration:.2f} s)")
    print(f"Throughput            : {len(tile_grid) / max(total_time, 1e-4):.1f} tiles/sec")
    print("\nGenerated Artifacts:")
    print(f"  GeoJSON Predictions : {saved_files['geojson']}")
    print(f"  CSV Statistics      : {saved_files['csv']}")
    print(f"  Overlay Image       : {saved_files['overlay']}")
    print("=" * 75)


if __name__ == "__main__":
    main()
