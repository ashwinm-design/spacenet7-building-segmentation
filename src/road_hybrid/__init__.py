"""
src/road_hybrid/__init__.py

SIH26012 Phase 13D: Hybrid Road Fusion & Topological Access Network Package.
Integrates classical computer-vision structural priors with deep-learning road probabilities,
building exclusion constraints, and topological graph refinement.
"""

from .fusion import (
    normalize_probability_maps,
    fuse_roads_weighted,
    fuse_roads_dl_recovery,
    fuse_roads_union_filtered,
    apply_building_exclusion,
    refine_road_connectivity,
    run_fusion_strategy_experiments,
    HybridRoadResult,
)
from .graph import (
    build_hybrid_road_graph,
    classify_edge_source_evidence,
    vectorize_hybrid_roads,
)
from .evaluation import (
    evaluate_hybrid_road_structure,
    compare_all_road_methods,
)
from .visualization import (
    plot_probability_histograms,
    generate_all_phase13d_figures,
)
from .failure_cases import (
    extract_failure_and_agreement_cases,
    generate_failure_case_report,
)

__all__ = [
    "normalize_probability_maps",
    "fuse_roads_weighted",
    "fuse_roads_dl_recovery",
    "fuse_roads_union_filtered",
    "apply_building_exclusion",
    "refine_road_connectivity",
    "run_fusion_strategy_experiments",
    "HybridRoadResult",
    "build_hybrid_road_graph",
    "classify_edge_source_evidence",
    "vectorize_hybrid_roads",
    "evaluate_hybrid_road_structure",
    "compare_all_road_methods",
    "plot_probability_histograms",
    "generate_all_phase13d_figures",
    "extract_failure_and_agreement_cases",
    "generate_failure_case_report",
]
