"""
src/prototype/polygonization.py

Instance reconstruction (Class-1 + 1.0px boundary growth), polygonization,
cleaning, confidence estimation, GeoJSON export, and visualization overlay.
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import csv
import json
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Polygon, MultiPolygon, shape, mapping
from shapely.ops import transform as shapely_transform
import rasterio.features as rio_features
import geopandas as gpd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .config import InferenceConfig, DEFAULT_CONFIG
from .geospatial import GeospatialMetadata, transform_polygons_to_crs, calculate_polygon_metrics


def grow_interior_into_boundary(
    interior_labels: np.ndarray,
    boundary_mask: np.ndarray,
    max_growth_distance: float = 1.0
) -> np.ndarray:
    """Class-1 interior dilation into boundary mask using Euclidean distance transform."""
    if np.sum(interior_labels > 0) == 0 or np.sum(boundary_mask > 0) == 0 or max_growth_distance <= 0:
        return interior_labels.copy()

    dist, nearest_idx = distance_transform_edt(interior_labels == 0, return_indices=True)
    grown = interior_labels.copy()
    qualifying = (boundary_mask == 1) & (interior_labels == 0) & (dist <= max_growth_distance)
    grown[qualifying] = interior_labels[nearest_idx[0, qualifying], nearest_idx[1, qualifying]]
    return grown


def extract_building_polygons(
    pred_class: np.ndarray,
    building_prob: np.ndarray,
    config: Optional[InferenceConfig] = None
) -> Tuple[List[Polygon], List[Dict[str, Any]], np.ndarray]:
    """
    Performs boundary-aware instance extraction from 3-class prediction map.

    Returns:
        pixel_polygons: list of Shapely Polygons in pixel coordinates
        attributes: list of property dictionaries (id, confidence, min_conf, etc.)
        grown_labels: [H, W] instance label array
    """
    cfg = config or DEFAULT_CONFIG
    h, w = pred_class.shape[:2]

    interior_mask = (pred_class == 1).astype(np.uint8)
    boundary_mask = (pred_class == 2).astype(np.uint8)

    # 1. Connected components on Class 1 only
    num_labels, labels = cv2.connectedComponents(interior_mask)

    # 2. Boundary-aware instance growth (1.0 px)
    grown_labels = grow_interior_into_boundary(
        labels, boundary_mask, max_growth_distance=cfg.growth_distance
    )

    # 3. Polygonize instances
    labels_int32 = grown_labels.astype(np.int32)
    shapes_gen = rio_features.shapes(labels_int32, mask=(labels_int32 > 0))

    pixel_polygons = []
    attributes = []
    bldg_id = 1

    for geom, val in shapes_gen:
        if val <= 0:
            continue
        poly = shape(geom)
        
        # Clean geometry
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue

        # Flatten multi-polygons if needed
        poly_list = [poly] if poly.geom_type == "Polygon" else list(poly.geoms) if poly.geom_type == "MultiPolygon" else []

        for p in poly_list:
            if p.is_empty or p.area < cfg.min_area_pixels:
                continue

            # Simplify geometry
            if cfg.simplify_tolerance > 0:
                p_simp = p.simplify(cfg.simplify_tolerance, preserve_topology=True)
                if not p_simp.is_empty and p_simp.is_valid:
                    p = p_simp

            # Extract confidence statistics over polygon footprint
            minx, miny, maxx, maxy = map(int, [np.floor(p.bounds[0]), np.floor(p.bounds[1]), np.ceil(p.bounds[2]), np.ceil(p.bounds[3])])
            minx, miny = max(0, minx), max(0, miny)
            maxx, maxy = min(w, maxx), min(h, maxy)

            sub_w = maxx - minx
            sub_h = maxy - miny
            if sub_w <= 0 or sub_h <= 0:
                continue

            # Shift coords to local patch
            local_p = shapely_transform(lambda x, y: (np.asarray(x) - minx, np.asarray(y) - miny), p)
            mask_patch = np.zeros((sub_h, sub_w), dtype=np.uint8)
            try:
                cv2.fillPoly(mask_patch, [np.array(local_p.exterior.coords, dtype=np.int32)], 1)
            except Exception:
                mask_patch[:, :] = 1

            prob_patch = building_prob[miny:maxy, minx:maxx]
            sampled_probs = prob_patch[mask_patch == 1]
            if len(sampled_probs) > 0:
                mean_conf = float(np.mean(sampled_probs))
                min_conf = float(np.min(sampled_probs))
            else:
                mean_conf = 0.5
                min_conf = 0.5

            pixel_polygons.append(p)
            attributes.append({
                "id": bldg_id,
                "confidence": round(mean_conf, 4),
                "min_confidence": round(min_conf, 4),
            })
            bldg_id += 1

    return pixel_polygons, attributes, grown_labels


def build_footprint_geodataframe(
    pixel_polygons: List[Polygon],
    attributes: List[Dict[str, Any]],
    geo_meta: GeospatialMetadata
) -> gpd.GeoDataFrame:
    """
    Constructs a GeoDataFrame with metadata attributes and projected metric dimensions.
    """
    if len(pixel_polygons) == 0:
        return gpd.GeoDataFrame({
            "id": [],
            "area": [],
            "perimeter": [],
            "area_unit": [],
            "perimeter_unit": [],
            "centroid_x": [],
            "centroid_y": [],
            "confidence": [],
            "min_confidence": [],
            "is_georeferenced": [],
            "geometry": [],
        }, crs=geo_meta.crs if geo_meta.is_geotiff else None)

    # 1. Coordinate transformation
    if geo_meta.is_geotiff and geo_meta.transform is not None:
        geo_polys = transform_polygons_to_crs(pixel_polygons, geo_meta)
    else:
        geo_polys = pixel_polygons

    # 2. Attribute and metric computation
    rows = []
    for idx, (px_p, geo_p, attr) in enumerate(zip(pixel_polygons, geo_polys, attributes)):
        metrics = calculate_polygon_metrics(geo_p, geo_meta, pixel_poly=px_p)
        record = {
            "id": attr["id"],
            "area": metrics["area"],
            "perimeter": metrics["perimeter"],
            "area_unit": metrics["area_unit"],
            "perimeter_unit": metrics["perimeter_unit"],
            "centroid_x": metrics["centroid_x"],
            "centroid_y": metrics["centroid_y"],
            "confidence": attr["confidence"],
            "min_confidence": attr["min_confidence"],
            "is_georeferenced": geo_meta.is_geotiff,
            "geometry": geo_p,
        }
        rows.append(record)

    gdf = gpd.GeoDataFrame(rows, crs=geo_meta.crs if geo_meta.is_geotiff else None)
    return gdf


def save_inference_outputs(
    gdf: gpd.GeoDataFrame,
    rgb_image: np.ndarray,
    pixel_polygons: List[Polygon],
    output_basename: str,
    geo_meta: GeospatialMetadata,
    config: Optional[InferenceConfig] = None
) -> Dict[str, str]:
    """
    Saves predictions to GeoJSON, CSV statistics, and high-resolution PNG overlay.
    """
    cfg = config or DEFAULT_CONFIG
    os.makedirs(cfg.get_predictions_dir(), exist_ok=True)
    os.makedirs(cfg.get_statistics_dir(), exist_ok=True)
    os.makedirs(cfg.get_visualizations_dir(), exist_ok=True)

    geojson_path = os.path.join(cfg.get_predictions_dir(), f"{output_basename}_buildings.geojson")
    csv_path = os.path.join(cfg.get_statistics_dir(), f"{output_basename}_statistics.csv")
    overlay_path = os.path.join(cfg.get_visualizations_dir(), f"{output_basename}_overlay.png")

    # 1. Export GeoJSON
    if len(gdf) > 0:
        gdf.to_file(geojson_path, driver="GeoJSON")
    else:
        # Empty GeoJSON
        empty_fc = {"type": "FeatureCollection", "features": []}
        if geo_meta.is_geotiff and geo_meta.crs:
            empty_fc["crs"] = {"type": "name", "properties": {"name": str(geo_meta.crs)}}
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(empty_fc, f, indent=2)

    # 2. Export Statistics CSV
    stats_df = gdf.drop(columns=["geometry"], errors="ignore") if len(gdf) > 0 else gpd.GeoDataFrame()
    stats_df.to_csv(csv_path, index=False)

    # 3. Generate PNG Overlay Visualization
    generate_overlay_visualization(
        rgb_image,
        pixel_polygons,
        gdf,
        overlay_path,
        output_basename,
        geo_meta
    )

    return {
        "geojson": geojson_path,
        "csv": csv_path,
        "overlay": overlay_path,
    }


def generate_overlay_visualization(
    rgb_image: np.ndarray,
    pixel_polygons: List[Polygon],
    gdf: gpd.GeoDataFrame,
    output_path: str,
    scene_name: str,
    geo_meta: GeospatialMetadata,
    max_display_dim: int = 2048
):
    """Generates an aesthetic overlay of detected building footprints over the source image."""
    h, w = rgb_image.shape[:2]
    
    # Scale down if image is gigantic for smooth rendering and fast file saving
    scale = 1.0
    if max(h, w) > max_display_dim:
        scale = max_display_dim / float(max(h, w))
        disp_w, disp_h = int(w * scale), int(h * scale)
        display_img = cv2.resize(rgb_image, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    else:
        display_img = rgb_image.copy()

    fig, ax = plt.subplots(figsize=(12, 12), dpi=120)
    ax.imshow(display_img)

    # Draw building footprints
    cyan_edge = "#06b6d4"
    orange_edge = "#f97316"
    
    for i, p in enumerate(pixel_polygons):
        if p.is_empty:
            continue
        # Scale polygon coordinates to match display image
        coords = np.array(p.exterior.coords)
        if scale != 1.0:
            coords = coords * scale

        color = orange_edge if i % 2 == 0 else cyan_edge
        patch = matplotlib.patches.Polygon(
            coords,
            closed=True,
            fill=True,
            facecolor=color,
            alpha=0.35,
            edgecolor=color,
            linewidth=1.2,
        )
        ax.add_patch(patch)

    total_bldgs = len(pixel_polygons)
    total_area = float(gdf["area"].sum()) if total_bldgs > 0 else 0.0
    area_unit = gdf["area_unit"].iloc[0] if total_bldgs > 0 else "pixels^2"
    crs_str = str(geo_meta.crs) if geo_meta.is_geotiff and geo_meta.crs else "Non-georeferenced Pixel Space"

    title_str = (
        f"SIH26012 Prototype Inference: {scene_name}\n"
        f"Detected Buildings: {total_bldgs:,} | Total Area: {total_area:,.1f} {area_unit} | CRS: {crs_str}"
    )
    ax.set_title(title_str, fontsize=11, fontweight="bold", pad=10)
    ax.axis("off")

    legend_elements = [
        mpatches.Patch(facecolor=cyan_edge, edgecolor=cyan_edge, alpha=0.5, label="Building Footprint"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.85, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
