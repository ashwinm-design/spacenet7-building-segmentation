"""
backend/services/inference_service.py

Connects FastAPI requests to the existing Phase 12A inference engine in src/prototype/.
Loads model once, executes tiled segmentation, boundary-aware polygonization,
and exports all artifacts into results/prototype/api/<analysis_id>/.
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from rasterio.transform import Affine

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.prototype.config import InferenceConfig, DEFAULT_CONFIG
from src.prototype.geospatial import read_geospatial_metadata, GeospatialMetadata
from src.prototype.preprocessing import load_image_and_metadata, create_tile_grid
from src.prototype.inference import BuildingSegmentor
from src.prototype.polygonization import (
    extract_building_polygons,
    build_footprint_geodataframe,
    save_inference_outputs,
)
from backend.storage import storage

# Cached singleton instance of the frozen segmentor
_SEGMENTOR: Optional[BuildingSegmentor] = None


def get_segmentor(config: Optional[InferenceConfig] = None) -> BuildingSegmentor:
    """Returns or initializes the singleton BuildingSegmentor."""
    global _SEGMENTOR
    if _SEGMENTOR is None:
        cfg = config or DEFAULT_CONFIG
        _SEGMENTOR = BuildingSegmentor(cfg)
    return _SEGMENTOR


def run_analysis_pipeline(
    analysis_id: str,
    input_file_path: Path,
    roi: Optional[Tuple[int, int, int, int]] = None,
    config: Optional[InferenceConfig] = None,
) -> Dict[str, Any]:
    """
    Executes the end-to-end building footprint analysis for an upload session.

    Args:
        analysis_id: Unique job identifier.
        input_file_path: Path to the stored input image (GeoTIFF or RGB).
        roi: Optional bounding box [xmin, ymin, xmax, ymax] in pixels.
        config: Optional custom InferenceConfig.

    Returns:
        Structured summary metadata dictionary.
    """
    t_start = time.time()
    cfg = config or DEFAULT_CONFIG
    analysis_dir = storage.get_analysis_dir(analysis_id)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load image and metadata using existing Phase 12A preprocessing
    rgb_image, valid_mask, geo_meta = load_image_and_metadata(str(input_file_path))
    h, w = rgb_image.shape[:2]

    # Handle ROI crop
    if roi:
        xmin, ymin, xmax, ymax = roi
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)
        rgb_image = rgb_image[ymin:ymax, xmin:xmax]
        valid_mask = valid_mask[ymin:ymax, xmin:xmax]

        if geo_meta.is_geotiff and geo_meta.transform is not None:
            geo_meta.transform = geo_meta.transform @ Affine.translation(xmin, ymin)
            geo_meta.width = xmax - xmin
            geo_meta.height = ymax - ymin
        h, w = rgb_image.shape[:2]

    # 2. Tile grid
    tile_grid = create_tile_grid((h, w), cfg.tile_size, cfg.tile_stride)

    # 3. Model inference using cached segmentor
    segmentor = get_segmentor(cfg)
    t_inf_start = time.time()
    pred_class, building_prob, _ = segmentor.predict_full_image(rgb_image, valid_mask, tile_grid)
    inf_duration = time.time() - t_inf_start

    # 4. Instance reconstruction (Class-1 + 1.0px growth) & polygonization
    t_poly_start = time.time()
    pixel_polys, attributes, _ = extract_building_polygons(pred_class, building_prob, cfg)
    gdf = build_footprint_geodataframe(pixel_polys, attributes, geo_meta)
    poly_duration = time.time() - t_poly_start

    # 5. Save outputs directly into results/prototype/api/<analysis_id>/
    geojson_target = analysis_dir / "buildings.geojson"
    csv_target = analysis_dir / "statistics.csv"
    overlay_target = analysis_dir / "overlay.png"

    # Export GeoJSON (source CRS preserved)
    if len(gdf) > 0:
        gdf.to_file(str(geojson_target), driver="GeoJSON")
    else:
        empty_fc = {"type": "FeatureCollection", "features": []}
        if geo_meta.is_geotiff and geo_meta.crs:
            empty_fc["crs"] = {"type": "name", "properties": {"name": str(geo_meta.crs)}}
        with open(geojson_target, "w", encoding="utf-8") as f:
            json.dump(empty_fc, f, indent=2)

    # Export Statistics CSV
    stats_df = gdf.drop(columns=["geometry"], errors="ignore") if len(gdf) > 0 else gdf
    stats_df.to_csv(str(csv_target), index=False)

    # Export Overlay Visualization
    from src.prototype.polygonization import generate_overlay_visualization
    generate_overlay_visualization(
        rgb_image,
        pixel_polys,
        gdf,
        str(overlay_target),
        scene_name=f"analysis_{analysis_id}",
        geo_meta=geo_meta,
    )

    total_processing_time = round(time.time() - t_start, 2)
    bldg_count = len(gdf)
    tot_area = round(float(gdf["area"].sum()), 2) if bldg_count > 0 else 0.0
    avg_area = round(float(gdf["area"].mean()), 2) if bldg_count > 0 else 0.0
    avg_conf = round(float(gdf["confidence"].mean()), 4) if bldg_count > 0 else 0.0
    area_unit = gdf["area_unit"].iloc[0] if bldg_count > 0 else ("m^2" if geo_meta.is_geotiff else "pixels^2")

    metadata = {
        "analysis_id": analysis_id,
        "status": "completed",
        "building_count": bldg_count,
        "total_area": tot_area,
        "area_unit": area_unit,
        "average_building_area": avg_area,
        "average_confidence": avg_conf,
        "processing_time": total_processing_time,
        "inference_time": round(inf_duration, 2),
        "polygonization_time": round(poly_duration, 2),
        "tile_count": len(tile_grid),
        "crs": str(geo_meta.crs) if geo_meta.is_geotiff and geo_meta.crs else None,
        "georeferenced": geo_meta.is_geotiff,
        "image_width": w,
        "image_height": h,
        "geojson_url": f"/api/results/{analysis_id}/geojson",
        "statistics_url": f"/api/results/{analysis_id}/statistics",
        "overlay_url": f"/api/results/{analysis_id}/overlay",
    }

    # Save metadata.json
    storage.save_metadata(analysis_id, metadata)

    return metadata
