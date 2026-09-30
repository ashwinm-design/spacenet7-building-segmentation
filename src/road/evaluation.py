"""
src/road/evaluation.py

Quality and geometric metrics evaluation for extracted road corridors.
Quantifies corridor coverage, connectivity, geometric validity,
and structural building exclusion compliance.

IMPORTANT SCIENTIFIC INTEGRITY STATEMENT:
No authoritative road ground-truth layer was available in the Open Cities dataset;
therefore road accuracy (IoU, precision, recall) is NOT reported or fabricated.
"""

from typing import Dict, Any, Optional
import numpy as np
import geopandas as gpd
from shapely.ops import unary_union


def evaluate_road_extraction(
    roads_gdf: gpd.GeoDataFrame,
    buildings_gdf: Optional[gpd.GeoDataFrame] = None,
    roi_area_m2: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Computes quality and spatial integrity metrics for extracted road corridors.

    Args:
        roads_gdf: GeoDataFrame containing vectorized road corridors.
        buildings_gdf: Optional GeoDataFrame of building footprints.
        roi_area_m2: Total area of the Region of Interest in square meters.

    Returns:
        Structured dictionary of empirical road metrics.
    """
    n_components = len(roads_gdf)

    if n_components == 0:
        return {
            "road_component_count": 0,
            "total_extracted_road_area_m2": 0.0,
            "total_extracted_road_length_m": 0.0,
            "mean_estimated_width_m": 0.0,
            "median_estimated_width_m": 0.0,
            "valid_geometry_pct": 100.0,
            "roi_coverage_pct": 0.0,
            "building_intersection_count": 0,
            "building_intersection_area_m2": 0.0,
            "ground_truth_status": "No authoritative road ground-truth layer was available; therefore road accuracy is not reported.",
        }

    total_area = float(roads_gdf["area_m2"].sum())
    total_length = float(roads_gdf["length_m"].sum())
    mean_width = float(roads_gdf["width_estimate_m"].mean())
    median_width = float(roads_gdf["width_estimate_m"].median())
    valid_count = int(roads_gdf["geometry"].apply(lambda g: g.is_valid and not g.is_empty).sum())
    valid_pct = round(100.0 * valid_count / n_components, 2)

    coverage_pct = round(100.0 * total_area / roi_area_m2, 2) if roi_area_m2 and roi_area_m2 > 0 else None

    # Check structural building exclusion compliance
    bldg_inter_count = 0
    bldg_inter_area = 0.0

    if buildings_gdf is not None and len(buildings_gdf) > 0:
        roads_union = unary_union(roads_gdf.geometry.tolist())
        bldgs_union = unary_union(buildings_gdf.geometry.tolist())
        inter = roads_union.intersection(bldgs_union)
        if not inter.is_empty:
            bldg_inter_area = float(round(inter.area, 4))
            for _, b_row in buildings_gdf.iterrows():
                if b_row.geometry.intersects(roads_union):
                    bldg_inter_count += 1

    return {
        "road_component_count": n_components,
        "total_extracted_road_area_m2": round(total_area, 2),
        "total_extracted_road_length_m": round(total_length, 2),
        "mean_estimated_width_m": round(mean_width, 2),
        "median_estimated_width_m": round(median_width, 2),
        "valid_geometry_pct": valid_pct,
        "roi_coverage_pct": coverage_pct,
        "building_intersection_count": bldg_inter_count,
        "building_intersection_area_m2": bldg_inter_area,
        "ground_truth_status": "No authoritative road ground-truth layer was available; therefore road accuracy is not reported.",
    }
