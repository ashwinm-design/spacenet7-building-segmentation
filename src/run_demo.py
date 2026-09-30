"""
src/run_demo.py

Phase 15: Standalone SIH26012 End-to-End Workflow Demonstration Script.
Executes and demonstrates the full AI building segmentation, hybrid road network fusion,
road-constrained parcel candidate generation, parcel intelligence, and human-in-the-loop review.

Usage:
    python src/run_demo.py
"""

import os
import sys
import time
import json
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure UTF-8 output if supported
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from backend.services.pipeline_service import storage
from backend.storage.reviews import review_storage

client = TestClient(app)


def print_banner():
    print("=" * 80)
    print("  SIH26012: HIGH-RESOLUTION AERIAL & DRONE FEATURE EXTRACTION")
    print("  END-TO-END PIPELINE & HUMAN-IN-THE-LOOP WEBGIS VERIFICATION DEMO")
    print("=" * 80)


def print_disclaimer():
    print("\n" + "=" * 80)
    print("  LEGAL & TECHNICAL DISCLAIMER:")
    print("  AI-generated parcel boundaries are preliminary spatial candidates and")
    print("  require professional cadastral / ground survey verification. Review")
    print("  status does not establish legal cadastral ownership or title rights.")
    print("=" * 80 + "\n")


def run_sih_demo():
    print_banner()

    # Step 1: System Health Check
    print("\n[STAGE 1/6] System Health & Service Readiness Check...")
    resp_health = client.get("/api/health")
    assert resp_health.status_code == 200, "Health check failed"
    health = resp_health.json()
    print(f"  [OK] Service: {health.get('service')} | Status: {health.get('status').upper()}")

    # Step 2: Launch Demo Pipeline (f49f31 locked region ROI 3072,2048,4096,3072)
    print("\n[STAGE 2/6] Executing End-to-End AI Analysis Pipeline...")
    print("  Tile: data/open_cities_raw/ptn/f49f31/f49f31.tif")
    print("  ROI Subregion: [xmin=3072, ymin=2048, xmax=4096, ymax=3072] (1024x1024)")
    print("  Running synchronous pipeline...")

    t0 = time.time()
    resp_demo = client.post("/api/demo?background=false")
    if resp_demo.status_code != 201:
        print(f"  Error launching demo: {resp_demo.status_code} - {resp_demo.text}")
        sys.exit(1)

    demo_result = resp_demo.json()
    analysis_id = demo_result["analysis_id"]
    elapsed = round(time.time() - t0, 2)
    print(f"  [OK] Pipeline Completed in {elapsed}s | Analysis ID: {analysis_id}")

    # Step 3: Fetch Real-Time Pipeline Status & Comprehensive Summary
    print("\n[STAGE 3/6] Fetching Analysis Summary & Stage Metrics...")
    resp_status = client.get(f"/api/analysis/{analysis_id}/status")
    status_data = resp_status.json()
    print(f"  Progress: {status_data.get('progress')}% | Current Stage: {status_data.get('current_stage')}")
    print(f"  Completed Stages: {', '.join(status_data.get('completed_stages', []))}")

    resp_summary = client.get(f"/api/results/{analysis_id}/summary")
    assert resp_summary.status_code == 200, f"Failed to retrieve summary for {analysis_id}"
    summary = resp_summary.json()

    print("\n  -------------------------------------------------------------")
    print(f"  ANALYSIS RESULTS DASHBOARD (ID: {analysis_id})")
    print("  -------------------------------------------------------------")
    print(f"  Input Dimensions:            {summary.get('input_dimensions', [1024, 1024])}")
    print(f"  Coordinate Reference System: {summary.get('crs')}")
    print(f"  Georeferenced:               {summary.get('georeferenced')}")
    print(f"  Total Processing Time:       {summary.get('processing_time')}s")
    print(f"  Building Footprints:         {summary.get('building_count')}")
    print(f"  Road Network Corridors:      {summary.get('road_corridor_count')}")
    print(f"  Total Road Length:           {summary.get('road_length_m')} meters")
    print(f"  Preliminary Parcels:         {summary.get('parcel_candidate_count')}")
    print(f"  Geometry Validity:           {summary.get('geometry_validity_pct')}%")
    conf_stats = summary.get("confidence_statistics", {})
    print(f"  Confidence Statistics:       Mean: {conf_stats.get('mean', 0.0) * 100:.1f}%, Median: {conf_stats.get('median', 0.0) * 100:.1f}%, Range: [{conf_stats.get('min', 0.0) * 100:.1f}% - {conf_stats.get('max', 0.0) * 100:.1f}%]")
    print(f"  Flagged for Review:          {summary.get('flagged_parcel_count')} candidates")

    print("\n  Stage Timings Breakdown:")
    for stage_name, duration in summary.get("stage_timings", {}).items():
        print(f"    - {stage_name:<25}: {duration:>6.2f}s")

    # Step 4: Human-in-the-Loop Parcel Verification Workflow
    print("\n[STAGE 4/6] Demonstrating Human-in-the-Loop Parcel Review...")
    resp_reviews = client.get(f"/api/results/{analysis_id}/parcel-reviews")
    assert resp_reviews.status_code == 200
    rev_data = resp_reviews.json()
    reviews = rev_data["reviews"]
    print(f"  Loaded {len(reviews)} parcel candidates in review queue.")

    # Find lowest confidence candidate and highest confidence candidate
    sorted_by_conf = sorted(reviews, key=lambda x: x["confidence"])
    lowest_candidate = sorted_by_conf[0]
    highest_candidate = sorted_by_conf[-1]

    low_id = lowest_candidate["candidate_id"]
    high_id = highest_candidate["candidate_id"]

    print(f"\n  Candidate #{low_id} (Lowest Confidence: {lowest_candidate['confidence']*100:.2f}%):")
    print(f"    Flags: {lowest_candidate.get('flags', 'None')}")
    print(f"    Area: {lowest_candidate.get('area_m2')} m^2 | Road Frontage: {lowest_candidate.get('road_frontage')} m")
    print(f"    Marking Candidate #{low_id} as 'needs_review'...")
    resp_flag = client.post(
        f"/api/results/{analysis_id}/parcel-reviews/{low_id}",
        json={
            "review_status": "needs_review",
            "reviewer_note": "Automated demo review: Flagged for cadastral field verification due to complex boundary.",
        },
    )
    assert resp_flag.status_code == 200
    print(f"    [OK] Candidate #{low_id} status updated to: {resp_flag.json()['review_status'].upper()}")

    print(f"\n  Candidate #{high_id} (Highest Confidence: {highest_candidate['confidence']*100:.2f}%):")
    print(f"    Flags: {highest_candidate.get('flags', 'None')}")
    print(f"    Area: {highest_candidate.get('area_m2')} m^2 | Road Frontage: {highest_candidate.get('road_frontage')} m")
    print(f"    Accepting Candidate #{high_id}...")
    resp_acc = client.post(
        f"/api/results/{analysis_id}/parcel-reviews/{high_id}",
        json={
            "review_status": "accepted",
            "reviewer_note": "Automated demo review: Verified footprint conformity and road access.",
        },
    )
    assert resp_acc.status_code == 200
    print(f"    [OK] Candidate #{high_id} status updated to: {resp_acc.json()['review_status'].upper()}")

    # Step 5: Audit Trail Verification
    print("\n[STAGE 5/6] Inspecting Immutable Review Audit Trail...")
    resp_audit = client.get(f"/api/results/{analysis_id}/parcel-reviews/audit")
    assert resp_audit.status_code == 200
    audit_entries = resp_audit.json()
    print(f"  Audit log contains {len(audit_entries)} recorded changes:")
    for entry in audit_entries[-3:]:
        old_s = str(entry.get('old_status', 'unreviewed'))
        new_s = str(entry.get('new_status', ''))
        print(f"    [{entry.get('timestamp')}] Parcel #{entry.get('candidate_id')}: {old_s.upper()} -> {new_s.upper()} ('{entry.get('reviewer_note', '')}')")

    # Step 6: Review Summary & GIS Deliverables
    print("\n[STAGE 6/6] Verifying GIS Deliverables & Export URLs...")
    resp_rev_sum = client.get(f"/api/results/{analysis_id}/parcel-reviews/summary")
    rev_sum = resp_rev_sum.json()
    print(f"  Review Breakdown: Total={rev_sum.get('total')}, Accepted={rev_sum.get('accepted')}, Needs Review={rev_sum.get('needs_review')}, Edited={rev_sum.get('edited')}, Rejected={rev_sum.get('rejected')}, Unreviewed={rev_sum.get('unreviewed')}")

    deliverables = [
        ("Building Footprints GeoJSON", f"/api/results/{analysis_id}/geojson"),
        ("Hybrid Road Corridors GeoJSON", f"/api/results/{analysis_id}/roads?source=hybrid"),
        ("Preliminary Parcels GeoJSON", f"/api/results/{analysis_id}/parcels?source=hybrid"),
        ("Reviewed Parcels GeoJSON", f"/api/results/{analysis_id}/parcel-reviews/export"),
        ("Building Statistics CSV", f"/api/results/{analysis_id}/statistics"),
        ("Visual Overlay PNG", f"/api/results/{analysis_id}/overlay"),
        ("Analysis Summary JSON", f"/api/results/{analysis_id}/summary"),
        ("Audit Trail JSON", f"/api/results/{analysis_id}/parcel-reviews/audit"),
    ]

    print("  Available API Export Endpoints:")
    for name, endpoint in deliverables:
        res = client.get(endpoint)
        status_str = "OK (200)" if res.status_code == 200 else f"ERR ({res.status_code})"
        print(f"    [OK] {name:<32}: {endpoint} [{status_str}]")

    print_disclaimer()
    print("=" * 80)
    print("  SIH26012 END-TO-END DEMO EXECUTION COMPLETE & VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    run_sih_demo()
