"""
src/parcel/compound_and_vacant.py

Exploratory spatial analytics for SIH26012:
1. Multi-Building Compound Hypotheses:
   Identifies architectural clusters where multiple structures (e.g., primary residence,
   garages, sheds, outbuildings) share common access frontage without road bisubsection.
   Explicitly reported as a 'multi-building compound hypothesis', NOT a verified legal parcel.

2. Potential Vacant / Open Parcel Candidates:
   Identifies unbuilt, non-road interstitial spaces exhibiting direct road frontage
   and sufficient contiguous area (>= 50 m²).
   Explicitly labelled as 'Potential vacant/open parcel candidate', NOT a confirmed cadastral parcel.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid


def analyze_multi_building_compounds(
    buildings_gdf: gpd.GeoDataFrame,
    parcels_gdf: gpd.GeoDataFrame,
    roads_gdf: Optional[gpd.GeoDataFrame] = None,
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """
    Analyzes spatial clusters to identify plausible multi-building compounds.
    Hypothesis: In dense urban environments, a single property compound often comprises
    one primary structure and one or more auxiliary outbuildings situated within the same
    spatial cluster, unbisected by an observable road corridor.

    Returns:
        (GeoDataFrame of compound hypotheses, summary dictionary)
    """
    if "cluster_id" not in buildings_gdf.columns:
        from .clustering import cluster_buildings
        buildings_gdf, _ = cluster_buildings(buildings_gdf, distance_threshold=15.0)

    road_union = unary_union(roads_gdf.geometry.tolist()) if roads_gdf is not None and len(roads_gdf) > 0 else None

    compound_records = []
    compound_id = 1
    clusters = buildings_gdf.groupby("cluster_id")

    for cid, group in clusters:
        n_bldgs = len(group)
        if n_bldgs < 2:
            continue

        # Sort buildings by area (largest is primary structure candidate)
        sorted_group = group.sort_values(by="area", ascending=False) if "area" in group.columns else group
        primary_bldg = sorted_group.iloc[0]
        primary_id = int(primary_bldg["id"]) if "id" in primary_bldg else 0
        primary_area = float(primary_bldg.geometry.area)

        # Check if internal buffer connects them without crossing a road
        cluster_union = unary_union(group.geometry.tolist())
        hull_geom = cluster_union.convex_hull.buffer(4.0, quad_segs=2)

        # Check road intersection
        if road_union is not None and hull_geom.intersects(road_union):
            # A road crosses the convex hull; test individual components
            diff_hull = hull_geom.difference(road_union)
            if diff_hull.geom_type == "MultiPolygon":
                hull_geom = max(diff_hull.geoms, key=lambda g: g.intersection(primary_bldg.geometry).area)
            else:
                hull_geom = diff_hull

        # Identify which buildings in the cluster fall into this compound envelope
        contained_bldgs = [
            int(r["id"]) if "id" in r else i
            for i, r in group.iterrows()
            if hull_geom.intersects(r.geometry)
        ]

        if len(contained_bldgs) >= 2:
            compound_records.append({
                "compound_id": compound_id,
                "cluster_id": int(cid),
                "primary_building_id": primary_id,
                "building_count": len(contained_bldgs),
                "building_ids": str(contained_bldgs),
                "primary_area_m2": round(primary_area, 2),
                "total_footprint_m2": round(float(group.geometry.area.sum()), 2),
                "compound_envelope_m2": round(float(hull_geom.area), 2),
                "hypothesis_type": "multi_building_compound_hypothesis",
                "geometry": hull_geom,
            })
            compound_id += 1

    compounds_gdf = gpd.GeoDataFrame(compound_records, crs=buildings_gdf.crs)
    summary = {
        "multi_building_clusters_evaluated": int((buildings_gdf["cluster_id"].value_counts() > 1).sum()),
        "compounds_identified": len(compounds_gdf),
        "total_compound_buildings": int(compounds_gdf["building_count"].sum()) if len(compounds_gdf) > 0 else 0,
        "note": "Reported as multi-building compound hypothesis; does not establish legal parcel boundaries.",
    }
    return compounds_gdf, summary


def identify_vacant_land_candidates(
    roi_bounds: Tuple[float, float, float, float],
    buildings_gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame,
    parcels_gdf: gpd.GeoDataFrame,
    crs: Any,
    min_vacant_area_m2: float = 40.0,
    min_road_frontage_m: float = 3.0,
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """
    Identifies candidate vacant / open parcel areas.
    Residual unbuilt spaces outside building parcel setbacks that possess
    direct road corridor frontage.

    Returns:
        (GeoDataFrame of vacant parcel candidates, summary dictionary)
    """
    roi_box = box(*roi_bounds)
    road_union = unary_union(roads_gdf.geometry.tolist()) if len(roads_gdf) > 0 else Polygon()
    parcels_union = unary_union(parcels_gdf.geometry.tolist()) if len(parcels_gdf) > 0 else Polygon()
    bldgs_union = unary_union(buildings_gdf.geometry.tolist()) if len(buildings_gdf) > 0 else Polygon()

    # Residual open space = ROI - (parcels U roads U buildings)
    occupied_union = unary_union([road_union, parcels_union, bldgs_union])
    residual = roi_box.difference(occupied_union)

    vacant_records = []
    vacant_id = 1

    sub_geoms = residual.geoms if hasattr(residual, "geoms") else [residual]

    for geom in sub_geoms:
        if not isinstance(geom, Polygon) or geom.is_empty:
            continue

        clean_geom = make_valid(geom).simplify(0.4)
        if clean_geom.is_empty or clean_geom.area < min_vacant_area_m2:
            continue

        # Check road frontage
        frontage_m = 0.0
        if not road_union.is_empty:
            try:
                shared = clean_geom.boundary.intersection(road_union.buffer(0.08))
                frontage_m = float(shared.length)
            except Exception:
                frontage_m = 0.0

        if frontage_m >= min_road_frontage_m:
            vacant_records.append({
                "vacant_candidate_id": vacant_id,
                "area_m2": round(clean_geom.area, 2),
                "perimeter_m": round(clean_geom.length, 2),
                "road_frontage_m": round(frontage_m, 2),
                "candidate_type": "Potential vacant/open parcel candidate",
                "evidence_status": "Observable unbuilt land with direct road corridor frontage",
                "geometry": clean_geom,
            })
            vacant_id += 1

    vacant_gdf = gpd.GeoDataFrame(vacant_records, crs=crs)
    summary = {
        "vacant_candidates_identified": len(vacant_gdf),
        "total_vacant_area_m2": round(float(vacant_gdf["area_m2"].sum()), 2) if len(vacant_gdf) > 0 else 0.0,
        "mean_vacant_area_m2": round(float(vacant_gdf["area_m2"].mean()), 2) if len(vacant_gdf) > 0 else 0.0,
        "mean_road_frontage_m": round(float(vacant_gdf["road_frontage_m"].mean()), 2) if len(vacant_gdf) > 0 else 0.0,
        "disclaimer": "Labelled as Potential vacant/open parcel candidate; explicitly not confirmed cadastral parcels.",
    }
    return vacant_gdf, summary
