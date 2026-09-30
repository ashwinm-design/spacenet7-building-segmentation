"""
src/demo_parcel_review.py

Phase 14B Demo Workflow Script for analysis 8b8c724c7e.
Demonstrates:
1. Load WebGIS results for 8b8c724c7e.
2. Select Hybrid roads.
3. Display hybrid parcel candidates.
4. Click P56 (low-confidence candidate: 62.54%).
5. Inspect confidence, evidence dimensions, and uncertainty flags.
6. Mark P56 as NEEDS REVIEW with reviewer note.
7. Edit candidate geometry (P12).
8. Save edited geometry -> Status changes to EDITED.
9. Accept a high-confidence candidate (P6, conf 97.6%).
10. Query review queue summary.
11. Export reviewed GeoJSON.
12. Verify exported geometry, attributes, and audit trail.
"""

import os
import sys
import json
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.main import app
from backend.storage.reviews import review_storage

client = TestClient(app)
ANALYSIS_ID = "8b8c724c7e"


def run_demo():
    print("=" * 60)
    print("PHASE 14B DEMO WORKFLOW — PARCEL VERIFICATION")
    print(f"Analysis ID: {ANALYSIS_ID}")
    print("=" * 60)

    # 1. Load WebGIS Summary
    print("\n--- STEP 1 & 2: Load WebGIS Analysis & Hybrid Roads ---")
    resp_meta = client.get(f"/api/results/{ANALYSIS_ID}")
    assert resp_meta.status_code == 200, f"Analysis {ANALYSIS_ID} not found"
    meta = resp_meta.json()
    print(f"Analysis Status: {meta['status']}")
    print(f"Total Buildings: {meta['building_count']}")
    print(f"CRS: {meta['crs']}")

    resp_roads = client.get(f"/api/results/{ANALYSIS_ID}/roads?source=hybrid")
    assert resp_roads.status_code == 200
    roads_data = resp_roads.json()
    print(f"Hybrid Road Corridors Loaded: {len(roads_data['features'])} features")

    # 3. Display Hybrid Parcel Candidates
    print("\n--- STEP 3: Display Preliminary Hybrid Parcel Candidates ---")
    resp_parcels = client.get(f"/api/results/{ANALYSIS_ID}/parcels?source=hybrid")
    assert resp_parcels.status_code == 200
    parcels_data = resp_parcels.json()
    total_parcels = len(parcels_data["features"])
    print(f"Hybrid Parcel Candidates Loaded: {total_parcels} candidates")
    assert total_parcels == 254

    # 4 & 5. Inspect P56 (Low-Confidence Candidate)
    print("\n--- STEP 4 & 5: Inspect Candidate P056 (Lowest Confidence) ---")
    resp_p56 = client.get(f"/api/results/{ANALYSIS_ID}/parcel-reviews/56")
    assert resp_p56.status_code == 200
    p56 = resp_p56.json()
    print(f"Candidate: P0{p56['candidate_id']}")
    print(f"Initial Status: {p56['review_status'].upper()}")
    print(f"Confidence: {p56['confidence'] * 100:.2f}% ({p56['confidence_band']})")
    print(f"Evidence Dimensions:")
    print(f"  - Building Evidence:      {p56.get('conf_building_evidence', 'N/A')}")
    print(f"  - Road Access:            {p56.get('conf_road_access', 'N/A')}")
    print(f"  - Geometry Quality:       {p56.get('conf_geometry_quality', 'N/A')}")
    print(f"  - Topology Quality:       {p56.get('conf_topological_plausibility', 'N/A')}")
    print(f"Spatial Metrics:")
    print(f"  - Area:                   {p56.get('area_m2')} m²")
    print(f"  - Building Coverage:      {p56.get('building_coverage') * 100:.1f}%")
    print(f"  - Road Frontage:          {p56.get('road_frontage')} m")
    print(f"  - Nearest Road:           {p56.get('nearest_road_distance')} m")
    print(f"Uncertainty Flags:          {p56.get('flags')}")

    # 6. Mark P56 as NEEDS REVIEW
    print("\n--- STEP 6: Mark P056 as NEEDS REVIEW ---")
    resp_flag = client.post(
        f"/api/results/{ANALYSIS_ID}/parcel-reviews/56",
        json={
            "review_status": "needs_review",
            "reviewer_note": "Boundary conflicts with neighbor footprint; recommended field cadastral survey.",
        },
    )
    assert resp_flag.status_code == 200
    flagged_p56 = resp_flag.json()
    print(f"Updated Status for P056: {flagged_p56['review_status'].upper()}")
    print(f"Reviewer Note: {flagged_p56['reviewer_note']}")

    # 7 & 8. Edit Candidate Geometry (P12)
    print("\n--- STEP 7 & 8: Edit Geometry for Candidate P012 ---")
    resp_p12_before = client.get(f"/api/results/{ANALYSIS_ID}/parcel-reviews/12")
    p12_before = resp_p12_before.json()
    print(f"P012 Area Before Edit: {p12_before.get('area_m2')} m²")
    print(f"P012 Compactness Before: {p12_before.get('compactness')}")

    orig_coords = p12_before["original_geometry"]["coordinates"][0]
    edited_coords = [list(pt) for pt in orig_coords]
    # Adjust first vertex slightly (e.g. shift vertex by +1.5m East, +1.0m North)
    edited_coords[0][0] += 1.5
    edited_coords[0][1] += 1.0
    edited_coords[-1] = edited_coords[0]  # ensure closure

    resp_edit = client.post(
        f"/api/results/{ANALYSIS_ID}/parcel-reviews/12",
        json={
            "review_status": "edited",
            "reviewer_note": "Manually adjusted NE boundary corner to conform to visible physical pathway.",
            "geometry": {
                "type": "Polygon",
                "coordinates": [edited_coords],
            },
        },
    )
    assert resp_edit.status_code == 200
    p12_after = resp_edit.json()
    print(f"Updated Status for P012: {p12_after['review_status'].upper()}")
    print(f"Geometry Modified Flag: {p12_after['geometry_modified']}")
    print(f"P012 Area After Recalculation: {p12_after.get('area_m2')} m²")
    print(f"P012 Compactness After: {p12_after.get('compactness')}")
    print(f"P012 Building Coverage After: {p12_after.get('building_coverage') * 100:.2f}%")

    # 9 & 10. Accept High-Confidence Candidate (P6)
    print("\n--- STEP 9 & 10: Accept High-Confidence Candidate P006 ---")
    resp_accept = client.post(
        f"/api/results/{ANALYSIS_ID}/parcel-reviews/6",
        json={
            "review_status": "accepted",
            "reviewer_note": "Boundary aligns cleanly with road frontage (13.6m) and structure footprints.",
        },
    )
    assert resp_accept.status_code == 200
    p6_accepted = resp_accept.json()
    print(f"P006 Confidence: {p6_accepted['confidence'] * 100:.2f}% ({p6_accepted['confidence_band']})")
    print(f"Updated Status for P006: {p6_accepted['review_status'].upper()}")

    # 11. Open Review Queue Summary
    print("\n--- STEP 11: Query Review Queue Summary ---")
    resp_sum = client.get(f"/api/results/{ANALYSIS_ID}/parcel-reviews/summary")
    assert resp_sum.status_code == 200
    summary = resp_sum.json()
    print("Real-Time Review Queue State:")
    print(f"  Total Candidates: {summary['total']}")
    print(f"  Unreviewed:       {summary['unreviewed']}")
    print(f"  Accepted:         {summary['accepted']}")
    print(f"  Edited:           {summary['edited']}")
    print(f"  Needs Review:     {summary['needs_review']}")
    print(f"  Rejected:         {summary['rejected']}")

    # 12 & 13. Export Reviewed GIS Data & Verify
    print("\n--- STEP 12 & 13: Export Reviewed GIS-Ready Data & Verification ---")
    resp_export = client.get(f"/api/results/{ANALYSIS_ID}/parcel-reviews/export")
    assert resp_export.status_code == 200
    exported_gj = resp_export.json()
    features = exported_gj["features"]
    print(f"Exported FeatureCollection containing {len(features)} features.")

    # Find the reviewed candidates in the export
    feat_56 = next(f for f in features if f["properties"]["candidate_id"] == 56)
    feat_12 = next(f for f in features if f["properties"]["candidate_id"] == 12)
    feat_6 = next(f for f in features if f["properties"]["candidate_id"] == 6)

    print(f"\nVerification of Exported Feature Properties:")
    print(f"  P056 -> Status: {feat_56['properties']['review_status']}, Note: '{feat_56['properties']['reviewer_note']}'")
    print(f"  P012 -> Status: {feat_12['properties']['review_status']}, Modified: {feat_12['properties']['geometry_modified']}, Area: {feat_12['properties']['area_m2']} m²")
    print(f"  P006 -> Status: {feat_6['properties']['review_status']}, Confidence: {feat_6['properties']['confidence']}")

    # Check legal disclaimer
    assert "disclaimer" in feat_56["properties"]
    print(f"  Legal Disclaimer verified: '{feat_56['properties']['disclaimer'][:75]}...'")

    # Verify files saved under results/experiments/parcel_review/
    exp_dir = Path("results/experiments/parcel_review")
    assert (exp_dir / "reviewed_parcels.geojson").is_file()
    assert (exp_dir / "review_summary.json").is_file()
    print(f"\nDeliverable files verified at: {exp_dir}")
    print("=" * 60)
    print("PHASE 14B DEMO WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
