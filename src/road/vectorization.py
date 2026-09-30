"""
src/road/vectorization.py

GIS Vectorization of Road and Access Corridors.
Converts raster binary masks and probability fields into valid, attribute-rich
GeoJSON/GeoDataFrame polygons and centerlines in target CRS.
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import cv2
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.validation import make_valid
from shapely.affinity import affine_transform
import rasterio.features


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    """Computes morphological skeleton using pure OpenCV erosion/dilation."""
    img = (mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    size = np.size(img)

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if (size - cv2.countNonZero(img)) == size:
            break
    return (skel > 0).astype(np.uint8)


def estimate_corridor_dimensions(geom: Polygon) -> Tuple[float, float, float]:
    """
    Estimates area, effective length, and average width of a road corridor polygon.
    Returns:
        (area_m2, length_m, width_estimate_m)
    """
    area_m2 = geom.area
    if area_m2 <= 0:
        return 0.0, 0.0, 0.0

    # Minimum bounding box oriented along primary axis
    min_rect = geom.minimum_rotated_rectangle
    if min_rect.geom_type == "Polygon":
        x, y = min_rect.exterior.coords.xy
        edge_lengths = [
            np.hypot(x[i+1] - x[i], y[i+1] - y[i]) for i in range(len(x) - 1)
        ]
        if len(edge_lengths) >= 2:
            length_m = max(edge_lengths[0], edge_lengths[1])
            width_m = min(edge_lengths[0], edge_lengths[1])
            # Effective width estimate: area / length
            effective_width_m = area_m2 / max(1e-3, length_m)
            return float(round(area_m2, 2)), float(round(length_m, 2)), float(round(effective_width_m, 2))

    # Fallback to perimeter approximation
    length_m = geom.length / 2.0
    effective_width_m = area_m2 / max(1e-3, length_m)
    return float(round(area_m2, 2)), float(round(length_m, 2)), float(round(effective_width_m, 2))


def vectorize_roads(
    mask: np.ndarray,
    transform: Any,
    crs: Any,
    prob_map: Optional[np.ndarray] = None,
    min_area_m2: float = 15.0,
    simplify_tol: float = 0.3,
) -> gpd.GeoDataFrame:
    """
    Converts binary road raster mask into a GIS-ready GeoDataFrame of corridor polygons.

    Args:
        mask: 2D uint8 binary mask (1 = road, 0 = non-road).
        transform: Affine transform mapping pixel (col, row) to world coordinates.
        crs: Coordinate Reference System (e.g., 'EPSG:32732').
        prob_map: Optional 2D float32 probability array for mean confidence calculation.
        min_area_m2: Minimum area threshold in square meters to discard minor artifacts.
        simplify_tol: Douglas-Peucker simplification tolerance in meters.

    Returns:
        GeoDataFrame with columns:
        [road_id, source, confidence, area_m2, length_m, width_estimate_m, geometry_valid, geometry]
    """
    if mask is None or np.sum(mask) == 0:
        return gpd.GeoDataFrame(
            columns=[
                "road_id", "source", "confidence", "area_m2",
                "length_m", "width_estimate_m", "geometry_valid", "geometry"
            ],
            crs=crs,
        )

    # Extract polygon contours from raster using rasterio
    shapes_generator = rasterio.features.shapes(
        mask.astype(np.uint8),
        mask=(mask > 0),
        transform=transform,
        connectivity=8,
    )

    records = []
    road_id = 1

    for geom_dict, val in shapes_generator:
        if val != 1:
            continue

        raw_geom = shape(geom_dict)
        if raw_geom.is_empty:
            continue

        # Geometric cleanup and validation
        valid_geom = make_valid(raw_geom)
        if valid_geom.is_empty:
            continue

        # Handle potential MultiPolygons from repair
        sub_geoms = valid_geom.geoms if hasattr(valid_geom, "geoms") else [valid_geom]

        for poly in sub_geoms:
            if not isinstance(poly, Polygon) or poly.is_empty:
                continue

            if simplify_tol > 0:
                poly = poly.simplify(simplify_tol, preserve_topology=True)

            area_m2 = poly.area
            if area_m2 < min_area_m2:
                continue

            area_val, length_val, width_val = estimate_corridor_dimensions(poly)

            # Calculate confidence
            conf = 0.75
            if prob_map is not None:
                # Approximate confidence using polygon interior sample or mean
                # Rasterize single polygon back to mask
                try:
                    p_mask = rasterio.features.geometry_mask(
                        [poly],
                        out_shape=mask.shape,
                        transform=transform,
                        invert=True,
                    )
                    if np.sum(p_mask) > 0:
                        conf = float(np.mean(prob_map[p_mask]))
                except Exception:
                    conf = 0.75

            records.append({
                "road_id": road_id,
                "source": "aerial_cv_baseline",
                "confidence": round(conf, 4),
                "area_m2": area_val,
                "length_m": length_val,
                "width_estimate_m": width_val,
                "geometry_valid": poly.is_valid,
                "geometry": poly,
            })
            road_id += 1

    gdf = gpd.GeoDataFrame(records, crs=crs)
    return gdf


def extract_road_centerlines(
    mask: np.ndarray,
    transform: Any,
    crs: Any,
    min_length_m: float = 10.0,
    simplify_tol: float = 0.5,
) -> gpd.GeoDataFrame:
    """
    Extracts topological road centerlines via morphological skeletonization.
    
    Args:
        mask: 2D uint8 binary mask.
        transform: Affine transform.
        crs: Target CRS.
        min_length_m: Minimum line length in meters.
        simplify_tol: Simplification tolerance in meters.
        
    Returns:
        GeoDataFrame of LineString centerlines.
    """
    if mask is None or np.sum(mask) == 0:
        return gpd.GeoDataFrame(columns=["centerline_id", "length_m", "geometry"], crs=crs)

    # Medial skeletonization
    skel = _morphological_skeleton(mask)

    # Convert skeleton lines to shapes
    shapes_gen = rasterio.features.shapes(
        skel,
        mask=(skel > 0),
        transform=transform,
        connectivity=8,
    )

    records = []
    line_id = 1

    for geom_dict, val in shapes_gen:
        if val != 1:
            continue
        g = shape(geom_dict)
        if g.is_empty:
            continue
        
        # Convert polygonized 1px skeleton cells into line representation if needed
        # Or exterior boundary midline
        boundary = g.exterior if hasattr(g, "exterior") else None
        if boundary is not None and boundary.length >= min_length_m * 2:
            coords = list(boundary.coords)
            half = len(coords) // 2
            line_geom = LineString(coords[:half+1]).simplify(simplify_tol)
            if line_geom.length >= min_length_m:
                records.append({
                    "centerline_id": line_id,
                    "length_m": round(line_geom.length, 2),
                    "geometry": line_geom,
                })
                line_id += 1

    return gpd.GeoDataFrame(records, crs=crs)
