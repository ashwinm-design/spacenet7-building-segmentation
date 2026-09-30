"""
src/parcel/__init__.py

SIH26012: Preliminary Urban Parcel Candidate Generation Engine.
Infers AI-assisted preliminary parcel boundaries from building spatial relationships.

DISCLAIMER:
Outputs are 'Preliminary Parcel Candidates'. They are NOT legally authoritative
cadastral boundaries and do not imply property ownership.
"""

from .clustering import cluster_buildings, evaluate_clustering_thresholds
from .partitioning import partition_voronoi, partition_concave_hull
from .topology import validate_and_repair_parcels, check_overlaps_and_gaps
from .scoring import score_parcel_candidates
from .parcel_engine import ParcelCandidateEngine
from .compound_and_vacant import (
    analyze_multi_building_compounds,
    identify_vacant_land_candidates,
)

__all__ = [
    "cluster_buildings",
    "evaluate_clustering_thresholds",
    "partition_voronoi",
    "partition_concave_hull",
    "validate_and_repair_parcels",
    "check_overlaps_and_gaps",
    "score_parcel_candidates",
    "ParcelCandidateEngine",
    "analyze_multi_building_compounds",
    "identify_vacant_land_candidates",
]
