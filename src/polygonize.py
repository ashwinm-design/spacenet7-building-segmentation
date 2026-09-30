import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape, Polygon
import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image

def mask_to_polygons(mask_arr, min_area=25, simplify_tol=0.5):
    """
    Converts a binary mask array (0 or 1) into Shapely Polygons.
    Filters out tiny noise polygons below min_area.
    Simplifies vertices to produce crisp building boundary edges.
    """
    binary_mask = (mask_arr > 0).astype(np.int16)
    shapes = features.shapes(binary_mask, mask=(binary_mask == 1))

    polygons = []
    removed_tiny = 0
    invalid_count = 0

    for geom, value in shapes:
        if value == 1:
            poly = shape(geom)
            if not poly.is_valid:
                poly = poly.buffer(0)
                invalid_count += 1
            if poly.is_empty:
                continue

            if poly.geom_type == 'Polygon':
                if poly.area < min_area:
                    removed_tiny += 1
                    continue
                if simplify_tol > 0:
                    poly = poly.simplify(simplify_tol, preserve_topology=True)
                polygons.append(poly)
            elif poly.geom_type == 'MultiPolygon':
                for sub_poly in poly.geoms:
                    if sub_poly.area < min_area:
                        removed_tiny += 1
                        continue
                    if simplify_tol > 0:
                        sub_poly = sub_poly.simplify(simplify_tol, preserve_topology=True)
                    polygons.append(sub_poly)

    return polygons, removed_tiny, invalid_count

def analyze_polygon_quality(polys):
    """
    E: Detailed polygon quality analysis.
    Returns dict with counts of valid, invalid, self-intersecting,
    tiny, fragmented, holes, and merged buildings.
    """
    stats = {
        "total": len(polys),
        "valid": 0,
        "invalid": 0,
        "self_intersecting": 0,
        "has_holes": 0,
        "fragmented": 0,       # Multi-part polygons from a single prediction blob
    }
    for p in polys:
        if p.is_valid:
            stats["valid"] += 1
        else:
            stats["invalid"] += 1
        if not p.is_simple:
            stats["self_intersecting"] += 1
        if hasattr(p, "interiors") and len(list(p.interiors)) > 0:
            stats["has_holes"] += 1
    return stats


def polygonize_predictions(min_area=25, simplify_tol=0.5, num_samples=None):
    pred_dir = "results/predictions"
    poly_dir = "results/polygons"
    vis_dir = "results/visualizations"
    splits_file = "data/processed/splits/test.txt"

    os.makedirs(poly_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    with open(splits_file, "r") as f:
        pairs = [line.strip().split(",") for line in f.readlines() if line.strip()]

    if num_samples is not None:
        pairs = pairs[:num_samples]

    all_raw_gdfs = []
    all_clean_gdfs = []
    total_polys_raw = 0
    total_polys_clean = 0
    total_invalid = 0
    total_removed_tiny = 0
    total_self_intersecting = 0
    total_holes = 0
    total_merged = 0

    print(f"Polygonizing predicted masks (min_area={min_area} px^2, simplify_tol={simplify_tol})...")

    for idx, (img_path, _) in enumerate(pairs):
        filename = os.path.basename(img_path)
        pred_path = os.path.join(pred_dir, f"pred_{filename}")

        if not os.path.exists(pred_path):
            continue

        pred_img = Image.open(pred_path).convert("L")
        pred_arr = (np.array(pred_img) > 127).astype(np.uint8)

        # ── Raw polygons (before cleaning) ──
        raw_polys, _, raw_invalid = mask_to_polygons(pred_arr, min_area=0, simplify_tol=0)
        total_polys_raw += len(raw_polys)

        # ── Cleaned polygons (filtered + simplified) ──
        clean_polys, removed_tiny, clean_invalid = mask_to_polygons(
            pred_arr, min_area=min_area, simplify_tol=simplify_tol
        )
        total_polys_clean += len(clean_polys)
        total_removed_tiny += removed_tiny
        total_invalid += clean_invalid

        # ── Quality analysis per image ──
        qa = analyze_polygon_quality(clean_polys)
        total_self_intersecting += qa["self_intersecting"]
        total_holes += qa["has_holes"]

        # Detect likely merged buildings: polygons significantly larger than median
        if len(clean_polys) >= 3:
            areas = sorted([p.area for p in clean_polys])
            median_area = areas[len(areas) // 2]
            merged = sum(1 for p in clean_polys if p.area > median_area * 5)
            total_merged += merged

        # Save raw GeoJSON
        if raw_polys:
            raw_gdf = gpd.GeoDataFrame({
                "geometry": raw_polys,
                "building_id": [f"RAW_{idx+1:03d}_{i:03d}" for i in range(len(raw_polys))],
                "area_px": [p.area for p in raw_polys],
                "source_image": [filename] * len(raw_polys),
            })
            raw_gdf.to_file(os.path.join(poly_dir, f"raw_polygons_{idx+1:02d}.geojson"), driver="GeoJSON")
            all_raw_gdfs.append(raw_gdf)

        # Save cleaned GeoJSON
        if clean_polys:
            clean_gdf = gpd.GeoDataFrame({
                "geometry": clean_polys,
                "building_id": [f"BLDG_{idx+1:03d}_{i:03d}" for i in range(len(clean_polys))],
                "area_px": [p.area for p in clean_polys],
                "source_image": [filename] * len(clean_polys),
            })
            clean_gdf.to_file(os.path.join(poly_dir, f"clean_polygons_{idx+1:02d}.geojson"), driver="GeoJSON")
            all_clean_gdfs.append(clean_gdf)

            # Create visual overlay
            orig_img = Image.open(img_path).convert("RGB")
            fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=150)

            ax[0].imshow(pred_arr, cmap="magma")
            ax[0].set_title(f"Sample {idx+1}: Predicted Binary Raster Mask")
            ax[0].axis("off")

            ax[1].imshow(orig_img)
            for poly in clean_polys:
                if poly.is_empty:
                    continue
                x, y = poly.exterior.xy
                ax[1].plot(x, y, color="#00FFCC", linewidth=2.0)
                ax[1].fill(x, y, facecolor="#00FFCC", alpha=0.35)

            ax[1].set_title(f"Sample {idx+1}: Extracted Vector Polygons ({len(clean_polys)} buildings)")
            ax[1].axis("off")

            plt.suptitle(f"Raster-to-Vector Footprint Extraction — Sample {idx+1}", fontsize=12)
            plt.tight_layout()
            poly_vis_path = os.path.join(vis_dir, f"polygon_vector_overlay_{idx+1:02d}.png")
            plt.savefig(poly_vis_path, bbox_inches="tight")
            plt.close()

    # Save consolidated GeoJSONs
    if all_clean_gdfs:
        combined_clean = gpd.GeoDataFrame(pd_concat_gdfs(all_clean_gdfs), geometry="geometry")
        combined_clean.to_file(os.path.join(poly_dir, "all_clean_building_footprints.geojson"), driver="GeoJSON")
        print(f"Saved cleaned footprints to: {poly_dir}/all_clean_building_footprints.geojson")

    if all_raw_gdfs:
        combined_raw = gpd.GeoDataFrame(pd_concat_gdfs(all_raw_gdfs), geometry="geometry")
        combined_raw.to_file(os.path.join(poly_dir, "all_raw_building_footprints.geojson"), driver="GeoJSON")
        print(f"Saved raw footprints to:     {poly_dir}/all_raw_building_footprints.geojson")

    # ── E: Polygon quality report ──
    quality_report = {
        "total_raw_polygons": total_polys_raw,
        "total_clean_polygons": total_polys_clean,
        "valid_polygons": total_polys_clean - total_invalid,
        "invalid_polygons_repaired": total_invalid,
        "self_intersections": total_self_intersecting,
        "tiny_polygons_removed": total_removed_tiny,
        "polygons_with_holes": total_holes,
        "likely_merged_buildings": total_merged,
        "fragmented_predictions": total_polys_raw - total_polys_clean - total_removed_tiny,
    }

    import json
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/polygon_quality.json", "w") as f:
        json.dump(quality_report, f, indent=2)

    print("\n── E: Polygon Quality Report ──")
    for k, v in quality_report.items():
        print(f"  {k:35s}: {v}")

    return quality_report

def pd_concat_gdfs(gdf_list):
    import pandas as pd
    return pd.concat(gdf_list, ignore_index=True)

if __name__ == "__main__":
    polygonize_predictions()
