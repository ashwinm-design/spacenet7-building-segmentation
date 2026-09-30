"""
src/parcel/parcel_engine.py

Unified Preliminary Urban Parcel Candidate Generation Pipeline.
Integrates spatial graph clustering, geometric partitioning, topology repair,
and evidence-based candidate scoring.
"""

from typing import Optional, Dict, Any, Tuple
import time
import geopandas as gpd
from shapely.geometry import Polygon

from .clustering import cluster_buildings
from .partitioning import partition_voronoi, partition_concave_hull
from .topology import validate_and_repair_parcels, check_overlaps_and_gaps
from .scoring import score_parcel_candidates


class ParcelCandidateEngine:
    """
    Engine for generating AI-assisted preliminary parcel candidates
    from building footprint polygons.
    """

    def __init__(
        self,
        cluster_threshold_m: float = 15.0,
        max_setback_m: float = 8.0,
        method: str = "voronoi",
        road_geometry: Optional[Any] = None,
    ):
        """
        Args:
            cluster_threshold_m: Distance in meters to group adjacent buildings into clusters.
            max_setback_m: Maximum outer setback limit for building parcels.
            method: Spatial partitioning approach ('voronoi' or 'concave_hull').
            road_geometry: Optional road network layer for spatial clipping (Phase 13B placeholder).
        """
        self.cluster_threshold_m = cluster_threshold_m
        self.max_setback_m = max_setback_m
        self.method = method.lower()
        self.road_geometry = road_geometry

    def generate_candidates(
        self,
        buildings_gdf: gpd.GeoDataFrame,
    ) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
        """
        Executes the complete parcel candidate generation pipeline.
        
        Returns:
            (GeoDataFrame of parcel candidates, execution metadata dictionary)
        """
        t0 = time.time()

        # Step 1: Spatial Adjacency Graph Clustering
        clustered_bldgs, cluster_meta = cluster_buildings(
            buildings_gdf, distance_threshold=self.cluster_threshold_m
        )

        # Step 2: Geometric Spatial Partitioning
        if self.method == "concave_hull":
            raw_parcels = partition_concave_hull(
                clustered_bldgs,
                cluster_col="cluster_id",
                max_setback_m=self.max_setback_m,
                road_geometry=self.road_geometry,
            )
        else:
            raw_parcels = partition_voronoi(
                clustered_bldgs,
                cluster_col="cluster_id",
                max_setback_m=self.max_setback_m,
                road_geometry=self.road_geometry,
            )

        # Step 3: Topology Validation and Repair
        validated_parcels, topology_meta = validate_and_repair_parcels(raw_parcels)
        overlap_meta = check_overlaps_and_gaps(validated_parcels)

        # Step 4: Evidence Scoring
        scored_parcels = score_parcel_candidates(
            validated_parcels, buildings_gdf, road_geometry=self.road_geometry
        )

        # Assign sequential parcel ID
        scored_parcels["parcel_id"] = list(range(1, len(scored_parcels) + 1))
        
        # Order columns logically
        col_order = [
            "parcel_id",
            "building_id",
            "cluster_id",
            "method",
            "parcel_area",
            "parcel_perimeter",
            "building_area",
            "coverage_ratio",
            "containment_score",
            "compactness",
            "road_touches",
            "road_distance_m",
            "road_frontage_m",
            "road_evidence_score",
            "candidate_score",
            "road_evidence",
            "topology_status",
            "geometry",
        ]
        available_cols = [c for c in col_order if c in scored_parcels.columns]
        scored_parcels = scored_parcels[available_cols]

        elapsed = time.time() - t0

        road_touching_count = int(scored_parcels["road_touches"].sum()) if "road_touches" in scored_parcels.columns else 0
        road_adj_pct = round(100.0 * road_touching_count / max(1, len(scored_parcels)), 2)
        mean_frontage = round(float(scored_parcels["road_frontage_m"].mean()), 2) if "road_frontage_m" in scored_parcels.columns else 0.0

        metadata = {
            "execution_time_seconds": round(elapsed, 3),
            "input_buildings": len(buildings_gdf),
            "candidate_parcels_generated": len(scored_parcels),
            "partitioning_method": self.method,
            "cluster_threshold_m": self.cluster_threshold_m,
            "max_setback_m": self.max_setback_m,
            "road_constraint_applied": self.road_geometry is not None,
            "road_touching_count": road_touching_count,
            "road_adjacency_pct": road_adj_pct,
            "mean_road_frontage_m": mean_frontage,
            "clustering_summary": cluster_meta,
            "topology_summary": topology_meta,
            "overlap_summary": overlap_meta,
            "mean_candidate_score": round(float(scored_parcels["candidate_score"].mean()), 4),
            "median_candidate_score": round(float(scored_parcels["candidate_score"].median()), 4),
        }

        return scored_parcels, metadata
