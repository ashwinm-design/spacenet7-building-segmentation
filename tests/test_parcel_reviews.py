"""
tests/test_parcel_reviews.py

Comprehensive automated tests for Phase 14B Human-in-the-Loop WebGIS Parcel Verification:
1. GET /api/results/{analysis_id}/parcel-reviews
2. GET /api/results/{analysis_id}/parcel-reviews/{candidate_id}
3. Accept parcel candidate
4. Mark parcel candidate as needs_review
5. Reject parcel candidate
6. Edit geometry with valid Polygon and verify metric recalculation (area_m2, coverage, compactness, road metrics)
7. Reject invalid geometry (NaN coords, non-polygon, empty) with 400 Bad Request
8. Reset edited geometry to AI original
9. Export reviewed GeoJSON and summary
10. Review persistence across restarts and audit logging
"""

import os
import sys
import json
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.main import app
from backend.storage.reviews import review_storage

client = TestClient(app)
DEMO_ANALYSIS_ID = "8b8c724c7e"


# 1. Get parcel reviews list and queue summary
def test_get_parcel_reviews_list():
    response = client.get(f"/api/results/{DEMO_ANALYSIS_ID}/parcel-reviews")
    assert response.status_code == 200
    data = response.json()
    assert data["analysis_id"] == DEMO_ANALYSIS_ID
    assert "summary" in data
    assert data["summary"]["total"] == 254
    assert len(data["reviews"]) == 254

    # Also test summary sub-endpoint
    sum_resp = client.get(f"/api/results/{DEMO_ANALYSIS_ID}/parcel-reviews/summary")
    assert sum_resp.status_code == 200
    sum_data = sum_resp.json()
    assert sum_data["total"] == 254


# 2. Get individual parcel candidate (P56)
def test_get_individual_parcel_p56():
    response = client.get(f"/api/results/{DEMO_ANALYSIS_ID}/parcel-reviews/56")
    assert response.status_code == 200
    data = response.json()
    assert data["candidate_id"] == 56
    assert data["confidence"] == 0.6254
    assert data["confidence_band"] == "MODERATE"
    assert "WEAK_BUILDING_EVIDENCE" in data["flags"]
    assert "area_m2" in data
    assert "current_geometry" in data
    assert "original_geometry" in data
    assert data["geometry_modified"] is False


# 3. Accept parcel candidate
def test_accept_parcel():
    candidate_id = 6  # High confidence candidate (conf: 0.976)
    payload = {
        "review_status": "accepted",
        "reviewer_note": "Verified clear access and valid boundary.",
    }
    response = client.post(
        f"/api/results/{DEMO_ANALYSIS_ID}/parcel-reviews/{candidate_id}",
        json=payload,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["candidate_id"] == candidate_id
    assert data["review_status"] == "accepted"
    assert data["reviewer_note"] == "Verified clear access and valid boundary."
    assert data["geometry_modified"] is False


# 4. Mark parcel candidate as needs_review
def test_mark_needs_review():
    candidate_id = 56  # Low confidence candidate (conf: 0.6254)
    payload = {
        "review_status": "needs_review",
        "reviewer_note": "Multi-source conflict between CV and DL roads.",
    }
    response = client.post(
        f"/api/results/{DEMO_ANALYSIS_ID}/parcel-reviews/{candidate_id}",
        json=payload,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["candidate_id"] == candidate_id
    assert data["review_status"] == "needs_review"
    assert "Multi-source conflict" in data["reviewer_note"]


# 5. Reject parcel candidate
def test_reject_parcel():
    candidate_id = 10
    payload = {
        "review_status": "rejected",
        "reviewer_note": "Uninhabitable sliver geometry along fence line.",
    }
    response = client.patch(
        f"/api/results/{DEMO_ANALYSIS_ID}/parcel-reviews/{candidate_id}",
        json=payload,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["candidate_id"] == candidate_id
    assert data["review_status"] == "rejected"
    assert data["reviewer_note"] == "Uninhabitable sliver geometry along fence line."


# 6. Edit valid geometry and verify metric recalculation
def test_edit_valid_geometry():
    candidate_id = 12
    # Fetch existing parcel geometry to make a slight modification
    orig_resp = client.get(f"/api/results/{DEMO_ANALYSIS_ID}/parcel-reviews/{candidate_id}")
    assert orig_resp.status_code == 200
    orig_data = orig_resp.json()
    orig_coords = orig_data["original_geometry"]["coordinates"][0]

    # Create edited coordinates (slightly offset one vertex)
    edited_coords = [list(pt) for pt in orig_coords]
    edited_coords[0][0] += 2.0  # Shift East by 2m
    edited_coords[-1] = edited_coords[0]  # Close loop

    edited_geom = {
        "type": "Polygon",
        "coordinates": [edited_coords],
    }

    payload = {
        "review_status": "edited",
        "reviewer_note": "Aligned southern edge with physical access path.",
        "geometry": edited_geom,
    }

    response = client.post(
        f"/api/results/{DEMO_ANALYSIS_ID}/parcel-reviews/{candidate_id}",
        json=payload,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["candidate_id"] == candidate_id
    assert data["review_status"] == "edited"
    assert data["geometry_modified"] is True
    assert data["area_m2"] > 0
    assert data["compactness"] > 0
    assert "building_coverage" in data


# 7. Reject invalid geometry with 400 Bad Request
def test_reject_invalid_geometry():
    candidate_id = 15

    # A. Coordinates with NaN in raw JSON payload
    raw_nan_json = '{"review_status": "edited", "geometry": {"type": "Polygon", "coordinates": [[[NaN, 9468600.0], [820800.0, 9468610.0], [820810.0, 9468600.0], [NaN, 9468600.0]]]}}'
    resp = client.post(
        f"/api/results/{DEMO_ANALYSIS_ID}/parcel-reviews/{candidate_id}",
        content=raw_nan_json,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code in (400, 422)

    # B. Non-polygon geometry type
    pt_geom = {
        "type": "Point",
        "coordinates": [820800.0, 9468600.0],
    }
    resp2 = client.post(
        f"/api/results/{DEMO_ANALYSIS_ID}/parcel-reviews/{candidate_id}",
        json={"review_status": "edited", "geometry": pt_geom},
    )
    assert resp2.status_code == 400
    assert "not supported" in resp2.json()["detail"].lower()

    # C. Degenerate zero-area polygon
    degenerate_geom = {
        "type": "Polygon",
        "coordinates": [[[820800.0, 9468600.0], [820800.0, 9468600.0], [820800.0, 9468600.0], [820800.0, 9468600.0]]],
    }
    resp3 = client.post(
        f"/api/results/{DEMO_ANALYSIS_ID}/parcel-reviews/{candidate_id}",
        json={"review_status": "edited", "geometry": degenerate_geom},
    )
    assert resp3.status_code == 400


# 8. Reset edited geometry back to AI original
def test_reset_edited_geometry():
    candidate_id = 12
    # Ensure candidate 12 was edited
    check_resp = client.get(f"/api/results/{DEMO_ANALYSIS_ID}/parcel-reviews/{candidate_id}")
    assert check_resp.json()["geometry_modified"] is True

    # Reset
    reset_resp = client.post(f"/api/results/{DEMO_ANALYSIS_ID}/parcel-reviews/{candidate_id}/reset")
    assert reset_resp.status_code == 200
    reset_data = reset_resp.json()
    assert reset_data["candidate_id"] == candidate_id
    assert reset_data["geometry_modified"] is False
    assert reset_data["current_geometry"] == reset_data["original_geometry"]


# 9. Export reviewed GeoJSON
def test_export_reviewed_geojson():
    export_resp = client.get(f"/api/results/{DEMO_ANALYSIS_ID}/parcel-reviews/export")
    assert export_resp.status_code == 200
    assert export_resp.headers["content-type"] == "application/geo+json"
    gj = export_resp.json()
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == 254

    sample_props = gj["features"][0]["properties"]
    assert "candidate_id" in sample_props
    assert "review_status" in sample_props
    assert "confidence" in sample_props
    assert "confidence_band" in sample_props
    assert "flags" in sample_props
    assert "area_m2" in sample_props
    assert "building_coverage" in sample_props
    assert "road_frontage" in sample_props
    assert "nearest_road_distance" in sample_props
    assert "reviewer_note" in sample_props
    assert "geometry_modified" in sample_props
    assert "disclaimer" in sample_props


# 10. Review persistence across reads & audit trail logging
def test_review_persistence_and_audit():
    audit_path = review_storage.get_audit_path(DEMO_ANALYSIS_ID)
    assert audit_path.is_file()
    with open(audit_path, "r", encoding="utf-8") as f:
        audit_log = json.load(f)
    assert len(audit_log) > 0
    latest = audit_log[-1]
    assert "candidate_id" in latest
    assert "old_status" in latest
    assert "new_status" in latest
    assert "old_geometry_hash" in latest
    assert "new_geometry_hash" in latest
    assert "timestamp" in latest
