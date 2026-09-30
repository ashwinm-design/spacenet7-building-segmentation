"""
src/parcel/scoring.py

Evidence-based candidate scoring engine for preliminary parcel boundaries.
Computes geometric, spatial containment, coverage, and compactness metrics.

IMPORTANT:
This metric is an 'AI Candidate Score' measuring geometric and spatial plausibility.
It is explicitly NOT a legal confidence score or property certainty rating.
"""

from typing import Dict, Any, List
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon


def compute_polsby_popper_compactness(area: float, perimeter: float) -> float:
    """
    Computes Polsby-Popper compactness: 4 * pi * Area / Perimeter^2.
    Bounded in [0, 1], where 1.0 is a circle and standard rectangular parcels are ~0.5 - 0.75.
    """
    if perimeter <= 0 or area <= 0:
        return 0.0
    val = (4.0 * np.pi * area) / (perimeter ** 2)
    return float(np.clip(val, 0.0, 1.0))


def compute_coverage_score(building_area: float, parcel_area: float) -> float:
    """
    Evaluates realistic urban building footprint-to-parcel ratio (coverage).
    Typical urban parcels exhibit building footprint coverage between 15% and 65%.
    """
    if parcel_area <= 0:
        return 0.0
    ratio = building_area / parcel_area
    if 0.15 <= ratio <= 0.65:
        return 1.0
    elif ratio > 0.65:
        # Building occupies almost entire parcel
        return float(max(0.2, 1.0 - (ratio - 0.65) * 2.0))
    else:
        # Parcel is overly expansive compared to building
        return float(max(0.2, ratio / 0.15))


def score_parcel_candidates(
    parcels_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    road_geometry: Optional[Any] = None,
) -> gpd.GeoDataFrame:
    """
    Computes multi-criteria evidence score for each preliminary parcel candidate.
    
    Evidence Factors:
      1. Building Containment (30% with road, 35% without): Footprint geometrically enclosed.
      2. Footprint Coverage Ratio (25%): Plausible urban building-to-parcel area ratio.
      3. Geometry Compactness (20%): Regularity of boundary (Polsby-Popper).
      4. Cluster Coherence (15% with road, 20% without): Absence of boundary fragmentation.
      5. Road Adjacency / Frontage (10% when road present): Direct access to physical corridor.
    """
    from shapely.ops import unary_union

    gdf = parcels_gdf.copy()
    bldgs_lookup = {row.id if hasattr(row, "id") and row.id is not None else i: row.geometry for i, row in buildings_gdf.iterrows()}
    bldgs_area_lookup = {row.id if hasattr(row, "id") and row.id is not None else i: row.geometry.area for i, row in buildings_gdf.iterrows()}

    road_union = None
    if road_geometry is not None:
        if isinstance(road_geometry, (gpd.GeoDataFrame, gpd.GeoSeries)):
            road_union = unary_union(road_geometry.geometry.tolist())
        elif isinstance(road_geometry, (list, tuple)):
            road_union = unary_union(road_geometry)
        else:
            road_union = road_geometry
        if road_union is not None and road_union.is_empty:
            road_union = None

    containment_scores = []
    coverage_ratios = []
    coverage_scores = []
    compactness_scores = []
    road_touches_list = []
    road_dist_list = []
    road_frontage_list = []
    road_evidence_scores = []
    road_evidence_texts = []
    candidate_scores = []

    for idx, row in gdf.iterrows():
        p_geom = row.geometry
        b_id = row.building_id if "building_id" in row else idx

        b_geom = bldgs_lookup.get(b_id, None)
        b_area = bldgs_area_lookup.get(b_id, 0.0)
        p_area = p_geom.area if p_geom is not None and not p_geom.is_empty else 0.0
        p_perim = p_geom.length if p_geom is not None and not p_geom.is_empty else 0.0

        # 1. Building Containment
        if b_geom is not None and p_geom is not None and not p_geom.is_empty:
            if p_geom.contains(b_geom):
                cont_score = 1.0
            else:
                inter_area = p_geom.intersection(b_geom).area
                cont_score = float(np.clip(inter_area / max(1e-4, b_area), 0.0, 1.0))
        else:
            cont_score = 0.0

        # 2. Footprint Coverage Ratio
        cov_ratio = float(b_area / max(1e-4, p_area))
        cov_score = compute_coverage_score(b_area, p_area)

        # 3. Compactness
        comp_score = compute_polsby_popper_compactness(p_area, p_perim)

        # 4. Cluster Coherence
        status = row.get("topology_status", "valid")
        if status == "valid":
            coherence_score = 1.0
        elif "repaired" in str(status):
            coherence_score = 0.8
        else:
            coherence_score = 0.3

        # 5. Road Frontage & Adjacency
        if road_union is not None and p_geom is not None and not p_geom.is_empty:
            dist = float(p_geom.distance(road_union))
            touches = dist < 0.2
            if touches:
                road_dist = 0.0
                # Frontage: shared boundary between parcel boundary and road corridor
                try:
                    shared_line = p_geom.boundary.intersection(road_union.buffer(0.08))
                    frontage_m = float(shared_line.length)
                except Exception:
                    frontage_m = 0.0

                if frontage_m >= 4.0:
                    rd_score = 1.0
                    rd_text = f"direct_access (frontage: {frontage_m:.1f}m)"
                elif frontage_m > 0:
                    rd_score = 0.75
                    rd_text = f"narrow_frontage (frontage: {frontage_m:.1f}m)"
                else:
                    rd_score = 0.60
                    rd_text = "corridor_contact (0m frontage)"
            else:
                road_dist = round(dist, 2)
                frontage_m = 0.0
                if road_dist <= 10.0:
                    rd_score = 0.45
                    rd_text = f"setback_access (dist: {road_dist:.1f}m)"
                else:
                    rd_score = 0.20
                    rd_text = f"interior_landlocked (dist: {road_dist:.1f}m)"

            # Composite Score with Road Adjacency Evidence (explainable weights)
            total_score = (
                0.30 * cont_score +
                0.25 * cov_score +
                0.20 * comp_score +
                0.15 * coherence_score +
                0.10 * rd_score
            )
            road_touches_list.append(touches)
            road_dist_list.append(road_dist)
            road_frontage_list.append(round(frontage_m, 2))
            road_evidence_scores.append(round(rd_score, 4))
            road_evidence_texts.append(rd_text)
        else:
            # Baseline Phase 13A weights
            total_score = (
                0.35 * cont_score +
                0.25 * cov_score +
                0.20 * comp_score +
                0.20 * coherence_score
            )
            road_touches_list.append(False)
            road_dist_list.append(None)
            road_frontage_list.append(0.0)
            road_evidence_scores.append(None)
            road_evidence_texts.append("unavailable")

        containment_scores.append(round(cont_score, 4))
        coverage_ratios.append(round(cov_ratio, 4))
        coverage_scores.append(round(cov_score, 4))
        compactness_scores.append(round(comp_score, 4))
        candidate_scores.append(round(total_score, 4))

    gdf["building_area"] = [bldgs_area_lookup.get(row.building_id, 0.0) for _, row in gdf.iterrows()]
    gdf["containment_score"] = containment_scores
    gdf["coverage_ratio"] = coverage_ratios
    gdf["compactness"] = compactness_scores
    gdf["candidate_score"] = candidate_scores
    gdf["road_touches"] = road_touches_list
    gdf["road_distance_m"] = road_dist_list
    gdf["road_frontage_m"] = road_frontage_list
    if road_union is not None:
        gdf["road_evidence_score"] = road_evidence_scores
    gdf["road_evidence"] = road_evidence_texts

    return gdf
