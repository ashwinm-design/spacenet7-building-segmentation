"""
src/road/

Modular Road and Access Corridor Extraction Package.
Extracts physical transportation and access features from high-resolution aerial imagery
to constrain preliminary parcel candidates.

IMPORTANT:
Road extraction is formulated as a physical feature extraction problem,
NOT a legal right-of-way determination problem. Extracted boundaries do NOT
represent surveyed legal road limits or official rights-of-way.
"""

from .preprocessing import (
    compute_exg,
    compute_color_features,
    compute_local_variance,
    create_building_exclusion_mask,
)
from .extraction import (
    RoadExtractionConfig,
    extract_road_mask,
)
from .vectorization import (
    vectorize_roads,
    extract_road_centerlines,
)
from .evaluation import evaluate_road_extraction
from .visualization import plot_road_visualizations

__all__ = [
    "compute_exg",
    "compute_color_features",
    "compute_local_variance",
    "create_building_exclusion_mask",
    "RoadExtractionConfig",
    "extract_road_mask",
    "vectorize_roads",
    "extract_road_centerlines",
    "evaluate_road_extraction",
    "plot_road_visualizations",
]
