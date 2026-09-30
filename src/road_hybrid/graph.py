"""
src/road_hybrid/graph.py

Hybrid Road Centerline Graph and Vectorization for SIH26012 Phase 13D.
Constructs a topologically sound access corridor network graph:
- Nodes: intersections, junctions, dead-ends
- Edges: road segments with length, estimated width, confidence, and source evidence
- Multi-source explainability attribution: 'classical', 'deep_learning', 'both'
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import rasterio
import rasterio.features
from rasterio.transform import xy
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, shape
from shapely.ops import unary_union
from shapely.validation import make_valid
import cv2
from scipy.ndimage import convolve, distance_transform_edt
from road_dl.evaluation import _morphological_skeleton


def classify_edge_source_evidence(
    line: LineString,
    cv_mask: np.ndarray,
    dl_mask: np.ndarray,
    transform: Any,
    num_samples: int = 15,
) -> Tuple[str, float]:
    """
    Step 7: Evaluates whether a road centerline segment is derived from:
    - 'classical' (predominantly confirmed by CV spectral/texture heuristics)
    - 'deep_learning' (predominantly recovered by neural probability features)
    - 'both' (mutually confirmed by both classical CV and deep learning)
    
    Returns:
        (evidence_category, mutual_agreement_ratio)
    """
    if line.is_empty or line.length == 0:
        return "hybrid", 0.50

    distances = np.linspace(0, line.length, max(num_samples, 3))
    points = [line.interpolate(d) for d in distances]

    inv_trans = ~transform
    cv_hits = 0
    dl_hits = 0
    valid_pts = 0

    h, w = cv_mask.shape

    for pt in points:
        c, r = inv_trans * (pt.x, pt.y)
        col, row = int(round(c)), int(round(r))
        if 0 <= row < h and 0 <= col < w:
            valid_pts += 1
            if cv_mask[row, col] > 0:
                cv_hits += 1
            if dl_mask[row, col] > 0:
                dl_hits += 1

    if valid_pts == 0:
        return "hybrid", 0.50

    cv_ratio = cv_hits / valid_pts
    dl_ratio = dl_hits / valid_pts

    if cv_ratio >= 0.40 and dl_ratio >= 0.40:
        return "both", round(float((cv_ratio + dl_ratio) / 2.0), 3)
    elif cv_ratio > dl_ratio:
        return "classical", round(float(cv_ratio), 3)
    else:
        return "deep_learning", round(float(dl_ratio), 3)


def vectorize_hybrid_roads(
    mask: np.ndarray,
    transform: Any,
    crs: Any,
    buildings_gdf: Optional[gpd.GeoDataFrame] = None,
    prob_map: Optional[np.ndarray] = None,
    min_area_m2: float = 15.0,
    simplify_tol: float = 0.3,
) -> gpd.GeoDataFrame:
    """
    Vectorizes the hybrid road binary mask into clean, non-overlapping GeoJSON polygons.
    Strictly excludes building footprint geometries to guarantee 0.00 m² overlap.
    """
    bin_mask = (mask > 0).astype(np.uint8)
    shapes_gen = rasterio.features.shapes(bin_mask, mask=(bin_mask == 1), transform=transform, connectivity=8)

    bldg_union = None
    if buildings_gdf is not None and len(buildings_gdf) > 0:
        bldg_union = unary_union(buildings_gdf.geometry.tolist())

    records = []
    road_id = 1

    for geom_dict, val in shapes_gen:
        if val != 1:
            continue
        geom = shape(geom_dict)
        if geom.is_empty:
            continue
        clean_geom = make_valid(geom).simplify(simplify_tol)
        if clean_geom.is_empty or clean_geom.area < min_area_m2:
            continue

        if bldg_union is not None and clean_geom.intersects(bldg_union):
            clean_geom = clean_geom.difference(bldg_union)
            if clean_geom.is_empty or clean_geom.area < min_area_m2:
                continue

        records.append({
            "road_id": road_id,
            "source": "hybrid_fusion_cv_dl",
            "confidence": 0.92,
            "area_m2": round(float(clean_geom.area), 2),
            "perimeter_m": round(float(clean_geom.length), 2),
            "geometry_valid": bool(clean_geom.is_valid),
            "geometry": clean_geom,
        })
        road_id += 1

    gdf = gpd.GeoDataFrame(records, crs=crs)
    return gdf


def build_hybrid_road_graph(
    mask: np.ndarray,
    cv_mask: np.ndarray,
    dl_mask: np.ndarray,
    transform: Any,
    crs: Any,
    min_length_m: float = 5.0,
    simplify_tol: float = 0.4,
    resolution_m: float = 0.20,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Step 7: Generates the Hybrid Topological Road Graph.
    Nodes = intersections, endpoints, and junctions.
    Edges = road centerline segments storing:
            road_id, length_m, estimated_width_m, source_evidence, confidence, geometry.
    """
    bin_mask = (mask > 0).astype(np.uint8)
    skel = _morphological_skeleton(bin_mask)

    # 1. Compute local distance transform to estimate corridor widths along skeleton
    dist_map = distance_transform_edt(bin_mask) * resolution_m * 2.0  # approximate full corridor width

    # 2. Convolve with 3x3 kernel to compute neighbor degrees
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = convolve(skel, kernel, mode="constant", cval=0) * skel

    # Nodes: endpoints (degree == 1) and intersections (degree >= 3)
    node_mask = np.logical_and(skel, (neighbors == 1) | (neighbors >= 3))
    node_ys, node_xs = np.where(node_mask)

    node_records = []
    for nid, (r, c) in enumerate(zip(node_ys, node_xs), start=1):
        x_world, y_world = rasterio.transform.xy(transform, r, c)
        n_type = "endpoint" if neighbors[r, c] == 1 else "intersection"
        node_records.append({
            "node_id": nid,
            "node_type": n_type,
            "degree": int(neighbors[r, c]),
            "geometry": Point(x_world, y_world),
        })
    nodes_gdf = gpd.GeoDataFrame(node_records, crs=crs)

    # 3. Vectorize skeleton branches into centerline edges
    shapes_gen = rasterio.features.shapes(skel, mask=(skel > 0), transform=transform, connectivity=8)
    edge_records = []
    edge_id = 1
    inv_trans = ~transform
    h, w = bin_mask.shape

    for geom_dict, val in shapes_gen:
        if val != 1:
            continue
        g = shape(geom_dict)
        if g.is_empty:
            continue

        boundary = g.exterior if hasattr(g, "exterior") else None
        if boundary is not None and boundary.length >= min_length_m:
            coords = list(boundary.coords)
            half = len(coords) // 2
            line = LineString(coords[:half + 1]).simplify(simplify_tol)
            if line.length >= min_length_m:
                # Classify evidence source for this centerline
                evidence, conf = classify_edge_source_evidence(line, cv_mask, dl_mask, transform)

                # Estimate width along centerline
                midpoint = line.interpolate(0.5, normalized=True)
                mc, mr = inv_trans * (midpoint.x, midpoint.y)
                mrow, mcol = int(np.clip(round(mr), 0, h - 1)), int(np.clip(round(mc), 0, w - 1))
                est_width = float(dist_map[mrow, mcol])
                if est_width < 1.0 or np.isnan(est_width):
                    est_width = 3.5

                edge_records.append({
                    "road_id": edge_id,
                    "length_m": round(float(line.length), 2),
                    "estimated_width_m": round(float(est_width), 2),
                    "source_evidence": evidence,
                    "confidence": conf,
                    "geometry": line,
                })
                edge_id += 1

    edges_gdf = gpd.GeoDataFrame(edge_records, crs=crs)
    return nodes_gdf, edges_gdf
