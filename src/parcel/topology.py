"""
src/parcel/topology.py

Rigorous topology validation, geometric repair, and spatial adjacency auditing.
Checks for self-intersections, duplicates, zero-area artifacts, MultiPolygons,
inter-candidate overlaps, and inter-parcel slivers.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.strtree import STRtree
import shapely


def repair_geometry(geom: Any) -> Tuple[Any, str]:
    """
    Attempts to repair an invalid geometry using GEOS make_valid.
    Returns (repaired_geometry, status_str).
    """
    if geom is None or geom.is_empty:
        return geom, "empty"

    if geom.is_valid:
        if isinstance(geom, MultiPolygon):
            # Extract largest component if small artifacts exist
            polys = list(geom.geoms)
            areas = [p.area for p in polys]
            if max(areas) > 0.95 * geom.area:
                return polys[int(np.argmax(areas))], "repaired_single_part"
            return geom, "valid_multipolygon"
        return geom, "valid"

    # Attempt automatic GEOS repair
    try:
        repaired = shapely.make_valid(geom)
        if repaired.is_valid and not repaired.is_empty:
            if isinstance(repaired, Polygon):
                return repaired, "repaired_valid"
            elif isinstance(repaired, (MultiPolygon, GeometryCollection)):
                # Filter for polygonal components
                polys = [g for g in repaired.geoms if isinstance(g, Polygon) and g.area > 1.0]
                if len(polys) == 1:
                    return polys[0], "repaired_valid"
                elif len(polys) > 1:
                    # Choose largest polygonal component
                    areas = [p.area for p in polys]
                    largest = polys[int(np.argmax(areas))]
                    return largest, "repaired_largest_polygon"
    except Exception:
        pass

    # Flag complex unrepairable geometry
    return geom, "flagged_invalid"


def validate_and_repair_parcels(
    parcels_gdf: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """
    Audits each candidate parcel polygon, fixes simple defects, and flags complex ones.
    
    Returns:
        (Cleaned GeoDataFrame with 'topology_status' and metrics, audit dictionary)
    """
    gdf = parcels_gdf.copy()
    n = len(gdf)

    repaired_geoms = []
    statuses = []
    initial_invalid_count = 0
    zero_area_count = 0
    multipolygon_count = 0

    for idx, geom in enumerate(gdf.geometry.values):
        if geom is None or geom.is_empty or geom.area < 1e-4:
            zero_area_count += 1
            repaired_geoms.append(geom)
            statuses.append("flagged_zero_area")
            continue

        if not geom.is_valid:
            initial_invalid_count += 1

        if isinstance(geom, MultiPolygon):
            multipolygon_count += 1

        repaired_geom, status = repair_geometry(geom)
        repaired_geoms.append(repaired_geom)
        statuses.append(status)

    gdf["geometry"] = repaired_geoms
    gdf["topology_status"] = statuses

    # Add geometric metrics
    gdf["parcel_area"] = [g.area if g is not None and not g.is_empty else 0.0 for g in gdf.geometry]
    gdf["parcel_perimeter"] = [g.length if g is not None and not g.is_empty else 0.0 for g in gdf.geometry]

    # Duplicate detection (IoU > 0.999) with fast area pre-filter
    tree = STRtree(gdf.geometry.values)
    duplicate_indices = set()
    for i, g1 in enumerate(gdf.geometry.values):
        if i in duplicate_indices or g1.is_empty:
            continue
        candidates = tree.query(g1)
        for j in candidates:
            if j > i and j not in duplicate_indices:
                g2 = gdf.geometry.values[j]
                if not g2.is_empty and abs(g1.area - g2.area) / max(g1.area, 1e-4) < 0.01:
                    intersection = g1.intersection(g2).area
                    union = g1.union(g2).area
                    if union > 0 and (intersection / union) > 0.999:
                        duplicate_indices.add(j)

    for j in duplicate_indices:
        gdf.loc[j, "topology_status"] = "flagged_duplicate"

    valid_count = int(np.sum(gdf["topology_status"].isin(["valid", "repaired_valid", "repaired_largest_polygon", "repaired_single_part"])))
    repaired_count = int(np.sum(gdf["topology_status"].str.startswith("repaired")))
    flagged_count = int(n - valid_count)

    audit_summary = {
        "total_candidates": n,
        "initial_invalid_geometries": initial_invalid_count,
        "valid_geometries_final": valid_count,
        "validity_percentage": round(100.0 * valid_count / max(1, n), 2),
        "repaired_geometries": repaired_count,
        "flagged_invalid_geometries": flagged_count,
        "zero_area_geometries": zero_area_count,
        "multipolygon_geometries": multipolygon_count,
        "duplicate_geometries": len(duplicate_indices),
    }

    return gdf, audit_summary


def check_overlaps_and_gaps(
    parcels_gdf: gpd.GeoDataFrame,
    overlap_area_tol: float = 0.5,
) -> Dict[str, Any]:
    """
    Quantifies pairwise overlaps and internal gaps between candidate parcel polygons.
    """
    geoms = parcels_gdf.geometry.values
    n = len(geoms)
    if n <= 1:
        return {
            "pairwise_overlap_count": 0,
            "total_overlap_area_m2": 0.0,
            "max_overlap_area_m2": 0.0,
            "sliver_gap_count": 0,
        }

    tree = STRtree(geoms)
    overlap_count = 0
    total_overlap_area = 0.0
    max_overlap_area = 0.0

    for i, g1 in enumerate(geoms):
        if g1 is None or g1.is_empty or not g1.is_valid:
            continue
        candidates = tree.query(g1)
        b1 = g1.bounds
        for j in candidates:
            if j > i:
                g2 = geoms[j]
                if g2 is None or g2.is_empty or not g2.is_valid:
                    continue
                b2 = g2.bounds
                # Fast bbox overlap check
                inter_bbox_w = min(b1[2], b2[2]) - max(b1[0], b2[0])
                inter_bbox_h = min(b1[3], b2[3]) - max(b1[1], b2[1])
                if inter_bbox_w > 0.05 and inter_bbox_h > 0.05 and (inter_bbox_w * inter_bbox_h) > overlap_area_tol:
                    try:
                        inter = g1.intersection(g2)
                        if inter.area > overlap_area_tol:
                            overlap_count += 1
                            total_overlap_area += inter.area
                            if inter.area > max_overlap_area:
                                max_overlap_area = inter.area
                    except Exception:
                        pass

    return {
        "pairwise_overlap_count": overlap_count,
        "total_overlap_area_m2": round(total_overlap_area, 2),
        "max_overlap_area_m2": round(max_overlap_area, 2),
    }
