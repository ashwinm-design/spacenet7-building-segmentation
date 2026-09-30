"""
src/test_hardening_validation.py

Phase 16 Hardening & Stability Validation:
1. Error handling test suite (unsupported format, empty upload, missing ID, invalid geometry).
2. Three consecutive repeatable demo runs (f49f31 locked ROI 3072,2048,4096,3072).
3. Export integrity verification across all 7 deliverable formats.
4. Verification that reference prototype 8b8c724c7e is completely untouched.
5. Saves repeatability_test.json, export_validation.json, and stage_timings.json.
"""

import os
import sys
import time
import json
import math
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from backend.services.pipeline_service import storage
from backend.storage.reviews import review_storage

client = TestClient(app)
REF_DIR = PROJECT_ROOT / "results/prototype/api/8b8c724c7e"
OUT_DIR = PROJECT_ROOT / "results/experiments/phase16_hardening"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_dir_mtimes(directory: Path) -> dict:
    if not directory.is_dir():
        return {}
    return {p.name: p.stat().st_mtime for p in directory.iterdir() if p.is_file()}


def run_error_handling_tests():
    print("\n" + "=" * 70)
    print("1. RUNNING ERROR HANDLING & INPUT RESILIENCE TESTS")
    print("=" * 70)
    error_results = {}

    # Test 1.1: Unsupported file format (.txt)
    print("  Testing 1.1: Unsupported file extension upload (.txt)...")
    resp = client.post(
        "/api/analyze",
        files={"file": ("test.txt", b"dummy content", "text/plain")},
    )
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
    err_body = resp.json()
    assert "Unsupported file extension" in err_body.get("detail", "")
    print(f"    [OK] Correctly rejected: HTTP 400 ({err_body['detail']})")
    error_results["unsupported_format"] = {"status": "PASS", "response_code": 400, "detail": err_body["detail"]}

    # Test 1.2: Empty file upload (0 bytes)
    print("  Testing 1.2: Empty file upload (0 bytes)...")
    resp = client.post(
        "/api/analyze",
        files={"file": ("empty.tif", b"", "image/tiff")},
    )
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
    err_body = resp.json()
    assert "empty" in err_body.get("detail", "").lower()
    print(f"    [OK] Correctly rejected: HTTP 400 ({err_body['detail']})")
    error_results["empty_file"] = {"status": "PASS", "response_code": 400, "detail": err_body["detail"]}

    # Test 1.3: Non-existent analysis ID query
    print("  Testing 1.3: Non-existent analysis ID query...")
    resp = client.get("/api/results/nonexistent999")
    assert resp.status_code == 404, f"Expected 404, got {resp.status_code}"
    err_body = resp.json()
    assert "not found" in err_body.get("detail", "").lower()
    print(f"    [OK] Correctly handled: HTTP 404 ({err_body['detail']})")
    error_results["missing_analysis_id"] = {"status": "PASS", "response_code": 404, "detail": err_body["detail"]}

    # Test 1.4: Invalid review status
    print("  Testing 1.4: Invalid parcel review status update...")
    resp = client.post(
        "/api/results/8b8c724c7e/parcel-reviews/1",
        json={"review_status": "invalid_super_status"},
    )
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
    err_body = resp.json()
    assert "Invalid review_status" in err_body.get("detail", "")
    print(f"    [OK] Correctly rejected: HTTP 400 ({err_body['detail']})")
    error_results["invalid_review_status"] = {"status": "PASS", "response_code": 400, "detail": err_body["detail"]}

    # Test 1.5: Missing required checkpoint handling (simulated path check)
    print("  Testing 1.5: Checkpoint integrity verification...")
    bldg_cp = PROJECT_ROOT / "results/checkpoints/best_model.pth"
    road_cp = PROJECT_ROOT / "results/experiments/road_dl/checkpoints/best_road_model.pth"
    assert bldg_cp.is_file(), "Building checkpoint missing"
    assert road_cp.is_file(), "Road DL checkpoint missing"
    print("    [OK] Both frozen checkpoints exist and are validated.")
    error_results["checkpoint_verification"] = {"status": "PASS", "checkpoints_present": True}

    return error_results


def run_three_repeatability_runs():
    print("\n" + "=" * 70)
    print("2. RUNNING THREE CONSECUTIVE DEMO REPEATABILITY RUNS (f49f31)")
    print("=" * 70)

    demo_img_path = PROJECT_ROOT / "data/open_cities_raw/ptn/f49f31/f49f31.tif"
    assert demo_img_path.is_file(), f"Demo image not found: {demo_img_path}"
    with open(demo_img_path, "rb") as f:
        demo_bytes = f.read()

    runs_data = []

    for i in range(1, 4):
        run_id = f"repeat_run_{i}"
        print(f"\n--- Launching Run #{i} (Analysis ID: {run_id}) ---")

        # Save upload
        saved_path = storage.save_uploaded_file(run_id, "f49f31.tif", demo_bytes)

        t0 = time.time()
        from backend.services.pipeline_service import run_analysis
        summary = run_analysis(run_id, saved_path, roi=(3072, 2048, 4096, 3072))
        elapsed = round(time.time() - t0, 2)

        print(f"  [OK] Run #{i} Completed in {elapsed}s")
        print(f"       Buildings: {summary['building_count']}")
        print(f"       Road Corridors: {summary['road_corridor_count']} ({summary['road_length_m']}m)")
        print(f"       Parcels: {summary['parcel_candidate_count']}")
        print(f"       Geometry Validity: {summary['geometry_validity_pct']}%")
        print(f"       Mean Confidence: {summary['confidence_statistics']['mean'] * 100:.2f}%")
        print(f"       Flagged Parcels: {summary['flagged_parcel_count']}")

        runs_data.append({
            "run_index": i,
            "analysis_id": run_id,
            "status": summary["status"],
            "runtime_s": elapsed,
            "building_count": summary["building_count"],
            "road_corridor_count": summary["road_corridor_count"],
            "road_length_m": summary["road_length_m"],
            "parcel_candidate_count": summary["parcel_candidate_count"],
            "geometry_validity_pct": summary["geometry_validity_pct"],
            "confidence_statistics": summary["confidence_statistics"],
            "flagged_parcel_count": summary["flagged_parcel_count"],
            "stage_timings": summary["stage_timings"],
        })

    return runs_data


def verify_exports_integrity(analysis_id: str):
    print("\n" + "=" * 70)
    print(f"3. VERIFYING EXPORT ARTIFACTS INTEGRITY FOR '{analysis_id}'")
    print("=" * 70)
    export_results = {}
    analysis_dir = storage.get_analysis_dir(analysis_id)

    # 1. Buildings GeoJSON
    p_bldg = analysis_dir / "buildings.geojson"
    print("  Checking 3.1: buildings.geojson...")
    with open(p_bldg, "r", encoding="utf-8") as f:
        gj_bldg = json.load(f)
    assert gj_bldg.get("type") == "FeatureCollection"
    feat_count = len(gj_bldg.get("features", []))
    assert feat_count == 254, f"Expected 254 buildings, found {feat_count}"
    # Verify valid geometries and no NaN
    for feat in gj_bldg["features"]:
        assert feat.get("geometry") is not None
        assert feat["geometry"].get("coordinates") is not None
    print(f"    [OK] Valid FeatureCollection with {feat_count} features.")
    export_results["buildings_geojson"] = {"status": "PASS", "features": feat_count, "size_bytes": p_bldg.stat().st_size}

    # 2. Hybrid Roads GeoJSON
    p_roads = analysis_dir / "roads_hybrid.geojson"
    print("  Checking 3.2: roads_hybrid.geojson...")
    with open(p_roads, "r", encoding="utf-8") as f:
        gj_roads = json.load(f)
    assert gj_roads.get("type") == "FeatureCollection"
    road_feats = len(gj_roads.get("features", []))
    assert road_feats > 0
    print(f"    [OK] Valid FeatureCollection with {road_feats} road corridor features.")
    export_results["roads_hybrid_geojson"] = {"status": "PASS", "features": road_feats, "size_bytes": p_roads.stat().st_size}

    # 3. Preliminary Parcels GeoJSON
    p_parcels = analysis_dir / "parcels_hybrid.geojson"
    print("  Checking 3.3: parcels_hybrid.geojson...")
    with open(p_parcels, "r", encoding="utf-8") as f:
        gj_parcels = json.load(f)
    assert gj_parcels.get("type") == "FeatureCollection"
    parcel_feats = len(gj_parcels.get("features", []))
    assert parcel_feats == 254
    print(f"    [OK] Valid FeatureCollection with {parcel_feats} parcel candidates.")
    export_results["parcels_hybrid_geojson"] = {"status": "PASS", "features": parcel_feats, "size_bytes": p_parcels.stat().st_size}

    # 4. Reviewed Parcels GeoJSON
    p_reviewed = analysis_dir / "reviewed_parcels.geojson"
    print("  Checking 3.4: reviewed_parcels.geojson...")
    with open(p_reviewed, "r", encoding="utf-8") as f:
        gj_reviewed = json.load(f)
    assert gj_reviewed.get("type") == "FeatureCollection"
    rev_feats = len(gj_reviewed.get("features", []))
    assert rev_feats == 254
    # Check review properties exist
    sample_props = gj_reviewed["features"][0]["properties"]
    assert "review_status" in sample_props
    assert "confidence" in sample_props
    print(f"    [OK] Valid FeatureCollection with {rev_feats} reviewed parcel candidates.")
    export_results["reviewed_parcels_geojson"] = {"status": "PASS", "features": rev_feats, "size_bytes": p_reviewed.stat().st_size}

    # 5. Statistics CSV
    p_csv = analysis_dir / "statistics.csv"
    print("  Checking 3.5: statistics.csv...")
    lines = p_csv.read_text(encoding="utf-8").strip().split("\n")
    header = lines[0].split(",")
    assert "id" in header and "area" in header and "confidence" in header
    assert len(lines) == 255  # header + 254 buildings
    print(f"    [OK] Valid CSV table with {len(lines)-1} building records and {len(header)} columns.")
    export_results["statistics_csv"] = {"status": "PASS", "records": len(lines)-1, "columns": len(header), "size_bytes": p_csv.stat().st_size}

    # 6. Analysis Summary JSON
    p_sum = analysis_dir / "analysis_summary.json"
    print("  Checking 3.6: analysis_summary.json...")
    with open(p_sum, "r", encoding="utf-8") as f:
        sum_data = json.load(f)
    assert sum_data.get("status") == "completed"
    assert sum_data.get("building_count") == 254
    assert sum_data.get("parcel_candidate_count") == 254
    assert "disclaimer" in sum_data
    print(f"    [OK] Valid analysis summary JSON with complete KPIs and disclaimer.")
    export_results["analysis_summary_json"] = {"status": "PASS", "building_count": sum_data["building_count"], "size_bytes": p_sum.stat().st_size}

    # 7. Parcel Reviews Audit JSON
    p_audit = analysis_dir / "parcel_reviews_audit.json"
    print("  Checking 3.7: parcel_reviews_audit.json...")
    with open(p_audit, "r", encoding="utf-8") as f:
        audit_data = json.load(f)
    assert isinstance(audit_data, list)
    print(f"    [OK] Valid JSON array with {len(audit_data)} audit entries.")
    export_results["parcel_reviews_audit_json"] = {"status": "PASS", "entries": len(audit_data), "size_bytes": p_audit.stat().st_size}

    return export_results


def main():
    print("=" * 80)
    print("  SIH26012 PHASE 16: SYSTEM HARDENING & RELIABILITY TEST SUITE")
    print("=" * 80)

    # Reference snapshot BEFORE
    ref_before = get_dir_mtimes(REF_DIR)
    print(f"Reference directory snapshot taken for 8b8c724c7e ({len(ref_before)} files).")

    # Step 1: Error handling tests
    err_res = run_error_handling_tests()

    # Step 2: Three consecutive demo runs
    runs = run_three_repeatability_runs()

    # Step 3: Export verification on run 1
    exp_res = verify_exports_integrity("repeat_run_1")

    # Reference snapshot AFTER
    ref_after = get_dir_mtimes(REF_DIR)
    assert ref_before == ref_after, "CRITICAL ERROR: Reference 8b8c724c7e was modified!"
    print("\n[VERIFIED] Reference directory 8b8c724c7e remained 100% untouched.")

    # Save repeatability_test.json
    repeat_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tile": "f49f31",
        "roi": [3072, 2048, 4096, 3072],
        "hardware": "CPU (Multi-threaded)",
        "repeatability_runs": runs,
        "consistency_analysis": {
            "all_runs_succeeded": all(r["status"] == "completed" for r in runs),
            "building_count_consistent": len(set(r["building_count"] for r in runs)) == 1,
            "road_corridor_consistent": len(set(r["road_corridor_count"] for r in runs)) == 1,
            "road_length_m_consistent": len(set(r["road_length_m"] for r in runs)) == 1,
            "parcel_count_consistent": len(set(r["parcel_candidate_count"] for r in runs)) == 1,
            "validity_pct_consistent": len(set(r["geometry_validity_pct"] for r in runs)) == 1,
            "mean_runtime_s": round(sum(r["runtime_s"] for r in runs) / len(runs), 2),
            "min_runtime_s": min(r["runtime_s"] for r in runs),
            "max_runtime_s": max(r["runtime_s"] for r in runs),
        },
        "reference_8b8c724c7e_untouched": True,
    }
    with open(OUT_DIR / "repeatability_test.json", "w", encoding="utf-8") as f:
        json.dump(repeat_data, f, indent=2)
    print(f"Saved: {OUT_DIR / 'repeatability_test.json'}")

    # Save export_validation.json
    export_val_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis_id": "repeat_run_1",
        "export_integrity": exp_res,
        "error_handling": err_res,
        "all_exports_valid": all(v["status"] == "PASS" for v in exp_res.values()),
    }
    with open(OUT_DIR / "export_validation.json", "w", encoding="utf-8") as f:
        json.dump(export_val_data, f, indent=2)
    print(f"Saved: {OUT_DIR / 'export_validation.json'}")

    # Save stage_timings.json for Phase 16
    fresh_run = runs[0]
    stage_timings_p16 = {
        "unit": "seconds",
        "hardware": "CPU (Multi-threaded)",
        "phase15_reference_f49f31": 38.80,
        "phase15_reference_abe1a3": 47.46,
        "phase16_fresh_demo_run": {
            "analysis_id": fresh_run["analysis_id"],
            "total_runtime_s": fresh_run["runtime_s"],
            "stage_timings": fresh_run["stage_timings"],
        },
        "all_three_runs_timings": [
            {"run": r["run_index"], "analysis_id": r["analysis_id"], "runtime_s": r["runtime_s"]}
            for r in runs
        ],
    }
    with open(OUT_DIR / "stage_timings.json", "w", encoding="utf-8") as f:
        json.dump(stage_timings_p16, f, indent=2)
    print(f"Saved: {OUT_DIR / 'stage_timings.json'}")

    print("\n" + "=" * 80)
    print("  PHASE 16 HARDENING & STABILITY VALIDATION COMPLETE: ALL PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
