"""
src/evaluate_parcel_intelligence.py

Phase 14A — Parcel Intelligence & Explainable Confidence Audit
SIH26012

Orchestrates the complete audit pipeline:
  1. Load Phase 13D hybrid outputs (building footprints, roads, parcels, road graph)
  2. Assemble evidence for each candidate
  3. Compute deterministic confidence scores
  4. Generate uncertainty flags
  5. Produce explainability reports
  6. Generate publication-quality visualizations
  7. Output structured audit results (GeoJSON, JSON, Markdown)

Usage:
    python src/evaluate_parcel_intelligence.py

Inputs:
    results/experiments/road_hybrid/candidate_parcels_hybrid_constrained.geojson
    results/experiments/road_hybrid/roads_hybrid.geojson
    results/experiments/road_hybrid/road_graph_edges.geojson
    results/prototype/buildings.geojson  (or polygons output)

Outputs:
    results/experiments/parcel_intelligence/
        parcels_audited.geojson          — Enriched parcel GeoJSON with confidence fields
        audit_summary.json               — Full audit statistics
        audit_narrative.md               — Human-readable audit report
        01_confidence_distribution.png   — Histogram
        02_band_distribution.png         — Bar chart
        03_component_radar.png           — Radar chart
        04_flag_distribution.png         — Flag bar chart
        05_spatial_confidence.png        — Spatial heatmap
        06_top_bottom_comparison.png     — Top/Bottom comparison
"""

import sys
import os
import json
import time
from pathlib import Path

import numpy as np
import geopandas as gpd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from parcel_intelligence import (
    ParcelConfidenceEngine,
    EvidenceAssembler,
    UncertaintyFlagger,
    ExplainabilityReport,
)
from parcel_intelligence.visualization import (
    plot_confidence_distribution,
    plot_band_distribution,
    plot_component_radar,
    plot_flag_distribution,
    plot_spatial_confidence_map,
    plot_top_bottom_comparison,
)


# ─── Configuration ───
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "road_hybrid"
OUTPUT_DIR = PROJECT_ROOT / "results" / "experiments" / "parcel_intelligence"

PARCELS_PATH = RESULTS_DIR / "candidate_parcels_hybrid_constrained.geojson"
ROADS_PATH = RESULTS_DIR / "roads_hybrid.geojson"
ROAD_EDGES_PATH = RESULTS_DIR / "road_graph_edges.geojson"

# Building footprints: try prototype first, then polygons
BUILDINGS_CANDIDATES = [
    PROJECT_ROOT / "results" / "prototype" / "api" / "8b8c724c7e" / "buildings.geojson",
    PROJECT_ROOT / "results" / "prototype" / "buildings.geojson",
    PROJECT_ROOT / "results" / "polygons" / "buildings_3class.geojson",
    PROJECT_ROOT / "results" / "polygons" / "buildings.geojson",
]


def find_buildings_path() -> Path:
    """Locate the building footprints GeoJSON from available paths."""
    for p in BUILDINGS_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No building footprints found. Searched:\n" +
        "\n".join(f"  - {p}" for p in BUILDINGS_CANDIDATES)
    )


def load_data():
    """Loads all Phase 13D input data."""
    print("=" * 70)
    print("PHASE 14A — PARCEL INTELLIGENCE & EXPLAINABLE CONFIDENCE AUDIT")
    print("SIH26012")
    print("=" * 70)
    print()

    # Load parcels
    print(f"Loading parcels: {PARCELS_PATH.name}")
    parcels_gdf = gpd.read_file(str(PARCELS_PATH))
    print(f"  → {len(parcels_gdf)} candidates loaded")
    print(f"  → CRS: {parcels_gdf.crs}")

    # Load roads
    print(f"Loading roads: {ROADS_PATH.name}")
    roads_gdf = gpd.read_file(str(ROADS_PATH))
    print(f"  → {len(roads_gdf)} road polygons loaded")

    # Load road graph edges
    road_edges_gdf = None
    if ROAD_EDGES_PATH.exists():
        print(f"Loading road graph: {ROAD_EDGES_PATH.name}")
        road_edges_gdf = gpd.read_file(str(ROAD_EDGES_PATH))
        print(f"  → {len(road_edges_gdf)} road edges loaded")
    else:
        print("  → Road graph edges not found; skipping edge-level evidence")

    # Load buildings
    bldg_path = find_buildings_path()
    print(f"Loading buildings: {bldg_path.name}")
    buildings_gdf = gpd.read_file(str(bldg_path))
    print(f"  → {len(buildings_gdf)} building footprints loaded")

    print()
    return parcels_gdf, roads_gdf, road_edges_gdf, buildings_gdf


def run_audit(parcels_gdf, roads_gdf, road_edges_gdf, buildings_gdf):
    """Executes the complete Phase 14A audit pipeline."""

    t0 = time.time()

    # ── Step 1: Assemble Evidence ──
    print("Step 1: Assembling evidence for each candidate...")
    assembler = EvidenceAssembler(
        buildings_gdf=buildings_gdf,
        road_gdf=roads_gdf,
        road_edges_gdf=road_edges_gdf,
    )
    evidence_list = assembler.assemble_evidence(parcels_gdf)
    print(f"  → {len(evidence_list)} evidence records assembled")

    # ── Step 2: Compute Confidence Scores ──
    print("Step 2: Computing deterministic confidence scores...")
    engine = ParcelConfidenceEngine()
    confidence_results = engine.score_batch(evidence_list)
    distribution = engine.compute_distribution_summary(confidence_results)

    print(f"  → Mean confidence: {distribution['mean_confidence']:.4f}")
    print(f"  → Median confidence: {distribution['median_confidence']:.4f}")
    print(f"  → Band distribution: {distribution['band_distribution']}")

    # ── Step 3: Generate Uncertainty Flags ──
    print("Step 3: Generating uncertainty flags...")
    flagger = UncertaintyFlagger()
    all_flags = flagger.flag_batch(evidence_list, confidence_results)
    flag_summary = flagger.compute_flag_summary(all_flags)

    print(f"  → {flag_summary['candidates_with_flags']}/{flag_summary['total_candidates']} candidates flagged ({flag_summary['flag_rate_pct']:.1f}%)")
    print(f"  → {flag_summary['total_flags_raised']} total flags raised")
    print(f"  → Flag distribution: {flag_summary['flag_type_distribution']}")

    # ── Step 4: Generate Explainability Reports ──
    print("Step 4: Generating explainability reports...")
    reporter = ExplainabilityReport()

    explanations = []
    for i, (ev, cr, fl) in enumerate(zip(evidence_list, confidence_results, all_flags)):
        pid = ev.get("parcel_id", i + 1)
        explanation = reporter.explain_single(pid, ev, cr, fl)
        explanations.append(explanation)

    top_bottom = reporter.identify_top_bottom(explanations, n=10)
    print(f"  → Top-10 confidence range: [{top_bottom['top_n'][-1]['confidence']:.4f}, {top_bottom['top_n'][0]['confidence']:.4f}]")
    print(f"  → Bottom-10 confidence range: [{top_bottom['bottom_n'][-1]['confidence']:.4f}, {top_bottom['bottom_n'][0]['confidence']:.4f}]")

    # Generate population narrative
    narrative = reporter.generate_population_narrative(distribution, flag_summary, top_bottom)

    elapsed = time.time() - t0
    print(f"\n  → Audit completed in {elapsed:.2f}s")

    return {
        "evidence_list": evidence_list,
        "confidence_results": confidence_results,
        "distribution": distribution,
        "all_flags": all_flags,
        "flag_summary": flag_summary,
        "explanations": explanations,
        "top_bottom": top_bottom,
        "narrative": narrative,
        "elapsed_seconds": round(elapsed, 3),
    }


def save_outputs(parcels_gdf, audit_results):
    """Saves all Phase 14A outputs."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving outputs to: {OUTPUT_DIR}")

    confidence_results = audit_results["confidence_results"]
    all_flags = audit_results["all_flags"]
    explanations = audit_results["explanations"]

    # ── Enrich parcels GeoDataFrame ──
    gdf = parcels_gdf.copy()
    gdf["confidence"] = [r["confidence"] for r in confidence_results]
    gdf["confidence_band"] = [r["confidence_band"] for r in confidence_results]

    # Add top-level sub-scores
    for dim in ["geometry_quality", "building_evidence", "road_access", "topological_plausibility"]:
        gdf[f"conf_{dim}"] = [
            r["confidence_components"][dim]["score"] for r in confidence_results
        ]

    # Add flag count and flag list
    gdf["flag_count"] = [len(fl) for fl in all_flags]
    gdf["flags"] = [
        ",".join(f["flag_id"] for f in fl) if fl else "" for fl in all_flags
    ]
    gdf["recommendation"] = [e["recommendation"] for e in explanations]

    # Save enriched GeoJSON
    geojson_path = OUTPUT_DIR / "parcels_audited.geojson"
    gdf.to_file(str(geojson_path), driver="GeoJSON")
    print(f"  → {geojson_path.name} ({len(gdf)} features)")

    # ── Save audit summary JSON ──
    summary = {
        "phase": "14A",
        "title": "Parcel Intelligence & Explainable Confidence Audit",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "execution_time_seconds": audit_results["elapsed_seconds"],
        "input_summary": {
            "total_candidates": len(parcels_gdf),
            "crs": str(parcels_gdf.crs),
        },
        "confidence_distribution": audit_results["distribution"],
        "flag_summary": audit_results["flag_summary"],
        "top_bottom_cases": audit_results["top_bottom"],
        "weights": {
            "geometry_quality": 0.25,
            "building_evidence": 0.30,
            "road_access": 0.25,
            "topological_plausibility": 0.20,
        },
        "disclaimer": (
            "All confidence scores measure AI inference quality and geometric plausibility. "
            "They do NOT represent legal cadastral certainty or property ownership. "
            "All boundaries are preliminary AI-assisted candidates requiring professional survey validation."
        ),
    }

    summary_path = OUTPUT_DIR / "audit_summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  → {summary_path.name}")

    # ── Save narrative report ──
    narrative_path = OUTPUT_DIR / "audit_narrative.md"
    with open(str(narrative_path), "w") as f:
        f.write(audit_results["narrative"])
    print(f"  → {narrative_path.name}")

    # ── Save per-candidate explanations JSON ──
    explanations_path = OUTPUT_DIR / "parcel_explanations.json"
    with open(str(explanations_path), "w") as f:
        json.dump(audit_results["explanations"], f, indent=2, default=str)
    print(f"  → {explanations_path.name} ({len(audit_results['explanations'])} records)")

    return gdf


def generate_visualizations(parcels_gdf, audit_results, buildings_gdf, roads_gdf):
    """Generates all Phase 14A visualization figures."""

    print("\nGenerating visualizations (300 DPI)...")

    confidence_scores = [r["confidence"] for r in audit_results["confidence_results"]]
    distribution = audit_results["distribution"]
    flag_summary = audit_results["flag_summary"]
    top_bottom = audit_results["top_bottom"]

    # 1. Confidence Distribution Histogram
    path = str(OUTPUT_DIR / "01_confidence_distribution.png")
    plot_confidence_distribution(confidence_scores, path)
    print(f"  → 01_confidence_distribution.png")

    # 2. Band Distribution Bar Chart
    path = str(OUTPUT_DIR / "02_band_distribution.png")
    plot_band_distribution(distribution["band_distribution"], path)
    print(f"  → 02_band_distribution.png")

    # 3. Component Radar Chart
    mean_components = {}
    for dim in ["geometry_quality", "building_evidence", "road_access", "topological_plausibility"]:
        scores = [
            r["confidence_components"][dim]["score"]
            for r in audit_results["confidence_results"]
        ]
        mean_components[dim] = round(float(np.mean(scores)), 4)

    path = str(OUTPUT_DIR / "03_component_radar.png")
    plot_component_radar(mean_components, path)
    print(f"  → 03_component_radar.png")

    # 4. Flag Distribution Chart
    path = str(OUTPUT_DIR / "04_flag_distribution.png")
    plot_flag_distribution(flag_summary["flag_type_distribution"], path)
    print(f"  → 04_flag_distribution.png")

    # 5. Spatial Confidence Map
    path = str(OUTPUT_DIR / "05_spatial_confidence.png")
    plot_spatial_confidence_map(
        parcels_gdf, confidence_scores,
        buildings_gdf=buildings_gdf,
        road_gdf=roads_gdf,
        output_path=path,
    )
    print(f"  → 05_spatial_confidence.png")

    # 6. Top/Bottom Comparison
    path = str(OUTPUT_DIR / "06_top_bottom_comparison.png")
    plot_top_bottom_comparison(top_bottom, path)
    print(f"  → 06_top_bottom_comparison.png")


def print_final_summary(audit_results):
    """Prints the final audit summary to console."""
    dist = audit_results["distribution"]
    flags = audit_results["flag_summary"]
    tb = audit_results["top_bottom"]

    print()
    print("=" * 70)
    print("PHASE 14A — AUDIT COMPLETE")
    print("=" * 70)
    print()
    print(f"  Total candidates audited:     {dist['total_candidates']}")
    print(f"  Mean confidence:              {dist['mean_confidence']:.4f}")
    print(f"  Median confidence:            {dist['median_confidence']:.4f}")
    print(f"  Std. deviation:               {dist['std_confidence']:.4f}")
    print(f"  Min / Max:                    {dist['min_confidence']:.4f} / {dist['max_confidence']:.4f}")
    print()
    print("  Band Distribution:")
    for band in ["HIGH", "MODERATE", "LOW", "VERY_LOW"]:
        cnt = dist["band_distribution"].get(band, 0)
        pct = dist["band_percentages"].get(band, 0.0)
        print(f"    {band:12s}  {cnt:4d}  ({pct:5.1f}%)")
    print()
    print(f"  Candidates with flags:        {flags['candidates_with_flags']}/{flags['total_candidates']} ({flags['flag_rate_pct']:.1f}%)")
    print(f"  Total flags raised:           {flags['total_flags_raised']}")
    print(f"  Mean flags per candidate:     {flags['mean_flags_per_candidate']:.2f}")
    print()
    print("  Top-5 Candidates:")
    for item in tb["top_n"][:5]:
        print(f"    Parcel {item['parcel_id']:4d}  confidence={item['confidence']:.4f}  band={item['confidence_band']}")
    print()
    print("  Bottom-5 Candidates:")
    for item in tb["bottom_n"][:5]:
        print(f"    Parcel {item['parcel_id']:4d}  confidence={item['confidence']:.4f}  flags={item['flags']}")
    print()
    print("  DISCLAIMER: Confidence scores measure AI inference quality.")
    print("  They do NOT represent legal cadastral certainty.")
    print("=" * 70)


def main():
    """Main entry point for Phase 14A evaluation."""
    try:
        # Load data
        parcels_gdf, roads_gdf, road_edges_gdf, buildings_gdf = load_data()

        # Run audit
        audit_results = run_audit(parcels_gdf, roads_gdf, road_edges_gdf, buildings_gdf)

        # Save outputs
        enriched_gdf = save_outputs(parcels_gdf, audit_results)

        # Generate visualizations
        generate_visualizations(enriched_gdf, audit_results, buildings_gdf, roads_gdf)

        # Print final summary
        print_final_summary(audit_results)

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Ensure Phase 13D outputs exist before running Phase 14A.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
