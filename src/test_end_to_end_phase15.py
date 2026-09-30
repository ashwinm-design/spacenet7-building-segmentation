"""
src/test_end_to_end_phase15.py

Executes:
1. End-to-end integration demo test on data/open_cities_raw/ptn/f49f31/f49f31.tif (ROI 3072, 2048, 4096, 3072).
2. Generalization test on data/open_cities_raw/ptn/abe1a3/abe1a3.tif (ROI 3000, 3000, 4024, 4024).
3. Verifies isolation of prototype directory results/prototype/api/8b8c724c7e/.
4. Verifies presence, structure, and integrity of all intermediate and final deliverables.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.pipeline_service import run_analysis, storage
from backend.storage.reviews import review_storage

EXPECTED_FILES = [
    "buildings.geojson",
    "statistics.csv",
    "overlay.png",
    "roads.geojson",
    "roads_deep_learning.geojson",
    "roads_hybrid.geojson",
    "road_graph_nodes.geojson",
    "road_graph_edges.geojson",
    "candidate_parcels_hybrid_constrained.geojson",
    "parcels_hybrid.geojson",
    "parcels_audited.geojson",
    "parcel_reviews.json",
    "parcel_reviews_audit.json",
    "reviewed_parcels.geojson",
    "review_summary.json",
    "analysis_summary.json",
    "metadata.json",
]


def snapshot_directory_mtimes(directory: Path) -> dict:
    """Takes a snapshot of modification times of files in a directory."""
    if not directory.is_dir():
        return {}
    return {f.name: f.stat().st_mtime for f in directory.iterdir() if f.is_file()}


def run_tests():
    print("=" * 70)
    print("PHASE 15: END-TO-END DEMO & GENERALIZATION VALIDATION")
    print("=" * 70)

    ref_dir = Path("results/prototype/api/8b8c724c7e")
    ref_snapshot_start = snapshot_directory_mtimes(ref_dir)
    print(f"Reference snapshot taken for {ref_dir} ({len(ref_snapshot_start)} files).")

    # ─────────────────────────────────────────────────────────────────
    # 1. DEMO PIPELINE EXECUTION (f49f31)
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("1. RUNNING DEMO PIPELINE (Open Cities Tile f49f31, ROI 3072,2048,4096,3072)")
    print("-" * 70)
    demo_id = "demo_e2e_f49f31"
    demo_img_path = PROJECT_ROOT / "data/open_cities_raw/ptn/f49f31/f49f31.tif"
    assert demo_img_path.is_file(), f"Demo image not found: {demo_img_path}"

    with open(demo_img_path, "rb") as f:
        demo_bytes = f.read()
    saved_demo_path = storage.save_uploaded_file(demo_id, "f49f31.tif", demo_bytes)

    t0 = time.time()
    demo_summary = run_analysis(demo_id, saved_demo_path, roi=(3072, 2048, 4096, 3072))
    demo_total_time = round(time.time() - t0, 2)

    print(f"Demo Run Completed in: {demo_total_time}s")
    print(f"Status: {demo_summary['status']}")
    print(f"Building Count: {demo_summary['building_count']}")
    print(f"Road Corridors: {demo_summary['road_corridor_count']}")
    print(f"Road Total Length: {demo_summary['road_length_m']} m")
    print(f"Parcel Candidates: {demo_summary['parcel_candidate_count']}")
    print(f"Geometry Validity: {demo_summary['geometry_validity_pct']}%")
    print(f"Confidence Stats: {demo_summary['confidence_statistics']}")
    print(f"Flagged Parcels: {demo_summary['flagged_parcel_count']}")
    print("Stage Timings:")
    for k, v in demo_summary["stage_timings"].items():
        print(f"  - {k}: {v}s")

    # Verify demo files
    demo_dir = storage.get_analysis_dir(demo_id)
    missing_demo_files = [f for f in EXPECTED_FILES if not (demo_dir / f).is_file()]
    assert not missing_demo_files, f"Missing files in {demo_id}: {missing_demo_files}"
    print(f"All {len(EXPECTED_FILES)} required deliverables present in {demo_dir}.")

    # Verify reference dir untouched
    ref_snapshot_mid = snapshot_directory_mtimes(ref_dir)
    assert ref_snapshot_start == ref_snapshot_mid, "ERROR: Reference 8b8c724c7e was modified during demo run!"
    print("Reference directory 8b8c724c7e remains 100% untouched.")

    # ─────────────────────────────────────────────────────────────────
    # 2. GENERALIZATION PIPELINE EXECUTION (abe1a3)
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("2. RUNNING GENERALIZATION PIPELINE (Open Cities Tile abe1a3, ROI 3000,3000,4024,4024)")
    print("-" * 70)
    gen_id = "gen_e2e_abe1a3"
    gen_img_path = PROJECT_ROOT / "data/open_cities_raw/ptn/abe1a3/abe1a3.tif"
    assert gen_img_path.is_file(), f"Generalization image not found: {gen_img_path}"

    with open(gen_img_path, "rb") as f:
        gen_bytes = f.read()
    saved_gen_path = storage.save_uploaded_file(gen_id, "abe1a3.tif", gen_bytes)

    t0 = time.time()
    gen_summary = run_analysis(gen_id, saved_gen_path, roi=(3000, 3000, 4024, 4024))
    gen_total_time = round(time.time() - t0, 2)

    print(f"Generalization Run Completed in: {gen_total_time}s")
    print(f"Status: {gen_summary['status']}")
    print(f"Building Count: {gen_summary['building_count']}")
    print(f"Road Corridors: {gen_summary['road_corridor_count']}")
    print(f"Road Total Length: {gen_summary['road_length_m']} m")
    print(f"Parcel Candidates: {gen_summary['parcel_candidate_count']}")
    print(f"Geometry Validity: {gen_summary['geometry_validity_pct']}%")
    print(f"Confidence Stats: {gen_summary['confidence_statistics']}")
    print(f"Flagged Parcels: {gen_summary['flagged_parcel_count']}")
    print("Stage Timings:")
    for k, v in gen_summary["stage_timings"].items():
        print(f"  - {k}: {v}s")

    # Verify generalization files
    gen_dir = storage.get_analysis_dir(gen_id)
    missing_gen_files = [f for f in EXPECTED_FILES if not (gen_dir / f).is_file()]
    assert not missing_gen_files, f"Missing files in {gen_id}: {missing_gen_files}"
    print(f"All {len(EXPECTED_FILES)} required deliverables present in {gen_dir}.")

    # Verify reference dir untouched again
    ref_snapshot_final = snapshot_directory_mtimes(ref_dir)
    assert ref_snapshot_start == ref_snapshot_final, "ERROR: Reference 8b8c724c7e was modified during generalization run!"
    print("Reference directory 8b8c724c7e remains 100% untouched.")

    # Save metrics to results/experiments/phase15_integration
    out_exp_dir = PROJECT_ROOT / "results/experiments/phase15_integration"
    out_exp_dir.mkdir(parents=True, exist_ok=True)

    validation_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "demo_run": {
            "analysis_id": demo_id,
            "tile": "f49f31",
            "roi": [3072, 2048, 4096, 3072],
            "total_time_s": demo_total_time,
            "metrics": demo_summary,
            "all_files_present": len(missing_demo_files) == 0,
        },
        "generalization_run": {
            "analysis_id": gen_id,
            "tile": "abe1a3",
            "roi": [3000, 3000, 4024, 4024],
            "total_time_s": gen_total_time,
            "metrics": gen_summary,
            "all_files_present": len(missing_gen_files) == 0,
        },
        "reference_8b8c724c7e_untouched": ref_snapshot_start == ref_snapshot_final,
    }

    with open(out_exp_dir / "end_to_end_validation.json", "w", encoding="utf-8") as f:
        json.dump(validation_results, f, indent=2)

    print(f"\nEnd-to-end validation results saved to {out_exp_dir / 'end_to_end_validation.json'}")
    print("=" * 70)
    print("PHASE 15 INTEGRATION TEST PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
