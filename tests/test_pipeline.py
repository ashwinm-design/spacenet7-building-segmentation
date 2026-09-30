"""
tests/test_pipeline.py

Integration and unit test suite for SIH26012 Phase 15:
End-to-End Integration, Analysis Orchestrator, Pipeline Status, and Deliverables Export.

Tests:
1. Analysis creation (POST /api/analyze with background=True)
2. Invalid upload rejection (format, empty file, corrupted raster)
3. Pipeline status endpoint (GET /api/analysis/{analysis_id}/status)
4. Successful pipeline state transitions and stage weighting
5. Multi-layer result retrieval (buildings, roads, parcels)
6. Failed-stage error handling and reporting
7. Analysis summary generation & schema validation (GET /api/results/{analysis_id}/summary)
8. Analysis output directory isolation (no cross-contamination)
9. Deliverables export endpoints (GeoJSON, summary JSON, audit JSON)
10. Parcel review verification workflow compatibility (Phase 14B integration)
"""

import os
import sys
import json
import uuid
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from backend.storage import storage
from backend.storage.reviews import review_storage
from backend.services.pipeline_service import (
    STAGE_WEIGHTS,
    validate_input_imagery,
    get_pipeline_status,
    update_pipeline_status,
)

client = TestClient(app)

DEMO_ID = "8b8c724c7e"
SAMPLE_GEOTIFF = "data/open_cities_raw/ptn/f49f31/f49f31.tif"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Analysis Creation Test
# ─────────────────────────────────────────────────────────────────────────────
def test_analysis_creation():
    """Verify POST /api/analyze with background=True queues analysis immediately."""
    import cv2
    import numpy as np
    valid_png = cv2.imencode(".png", np.zeros((64, 64, 3), dtype=np.uint8))[1].tobytes()

    with patch("backend.main.run_analysis_async") as mock_async:
        mock_async.return_value = MagicMock()
        response = client.post(
            "/api/analyze",
            files={"file": ("test_upload.png", valid_png, "image/png")},
            data={"background": "true"},
        )

    assert response.status_code == 201
    data = response.json()
    assert "analysis_id" in data
    assert data["status"] == "queued"
    assert data["current_stage"] == "QUEUED"
    assert data["progress"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Invalid Upload Rejection Test
# ─────────────────────────────────────────────────────────────────────────────
def test_invalid_upload_handling():
    """Verify invalid extensions, empty files, and corrupted data are rejected."""
    # Unsupported extension
    resp_ext = client.post(
        "/api/analyze",
        files={"file": ("malicious.exe", b"binarycontent", "application/octet-stream")},
    )
    assert resp_ext.status_code == 400
    assert "Unsupported file extension" in resp_ext.json()["detail"]

    # Empty file
    resp_empty = client.post(
        "/api/analyze",
        files={"file": ("empty.tif", b"", "image/tiff")},
    )
    assert resp_empty.status_code == 400
    assert "empty" in resp_empty.json()["detail"].lower()

    # Corrupted image file
    resp_corrupt = client.post(
        "/api/analyze",
        files={"file": ("corrupt.png", b"not-a-valid-png-header", "image/png")},
    )
    assert resp_corrupt.status_code == 400
    assert "validation failed" in resp_corrupt.json()["detail"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Status Endpoint Test
# ─────────────────────────────────────────────────────────────────────────────
def test_pipeline_status_endpoint():
    """Verify GET /api/analysis/{id}/status returns real stage and progress."""
    test_id = f"test_stat_{uuid.uuid4().hex[:6]}"
    update_pipeline_status(test_id, "BUILDING_INFERENCE", status="running")

    resp = client.get(f"/api/analysis/{test_id}/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["analysis_id"] == test_id
    assert data["status"] == "running"
    assert data["current_stage"] == "BUILDING_INFERENCE"
    assert data["progress"] == STAGE_WEIGHTS["BUILDING_INFERENCE"]

    # Non-existent analysis
    resp_404 = client.get("/api/analysis/nonexistent_xyz/status")
    assert resp_404.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pipeline State Progression & Stage Weighting Test
# ─────────────────────────────────────────────────────────────────────────────
def test_pipeline_stage_weighting():
    """Verify monotonic progress percentage mapping across all pipeline stages."""
    stages = [
        "QUEUED",
        "VALIDATING",
        "PREPROCESSING",
        "BUILDING_INFERENCE",
        "BUILDING_POLYGONIZATION",
        "ROAD_EXTRACTION",
        "ROAD_FUSION",
        "PARCEL_GENERATION",
        "PARCEL_INTELLIGENCE",
        "READY_FOR_REVIEW",
        "COMPLETED",
    ]

    for i in range(len(stages) - 1):
        s_current = stages[i]
        s_next = stages[i + 1]
        assert STAGE_WEIGHTS[s_current] <= STAGE_WEIGHTS[s_next], f"Stage weights non-monotonic: {s_current} > {s_next}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Result Retrieval on Demo Analysis
# ─────────────────────────────────────────────────────────────────────────────
def test_result_retrieval_layers():
    """Verify retrieval of buildings, roads, and parcel candidates GeoJSON."""
    # 5A. Building footprints
    resp_bldg = client.get(f"/api/results/{DEMO_ID}/geojson")
    assert resp_bldg.status_code == 200
    bldg_fc = resp_bldg.json()
    assert bldg_fc["type"] == "FeatureCollection"
    assert len(bldg_fc["features"]) > 0

    # 5B. Hybrid roads
    resp_roads = client.get(f"/api/results/{DEMO_ID}/roads?source=hybrid")
    assert resp_roads.status_code == 200
    roads_fc = resp_roads.json()
    assert roads_fc["type"] == "FeatureCollection"
    assert len(roads_fc["features"]) > 0

    # 5C. Preliminary parcels
    resp_parcels = client.get(f"/api/results/{DEMO_ID}/parcels?source=hybrid")
    assert resp_parcels.status_code == 200
    parcels_fc = resp_parcels.json()
    assert parcels_fc["type"] == "FeatureCollection"
    assert len(parcels_fc["features"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Failed Stage Reporting Test
# ─────────────────────────────────────────────────────────────────────────────
def test_failed_stage_reporting():
    """Verify failure state stores current stage and human-readable error."""
    fail_id = f"fail_{uuid.uuid4().hex[:6]}"
    update_pipeline_status(
        analysis_id=fail_id,
        stage="ROAD_FUSION",
        status="failed",
        error="Hybrid road fusion failed: disconnected component graph",
        technical_error="RuntimeError('disconnected graph')",
    )

    resp = client.get(f"/api/analysis/{fail_id}/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed"
    assert data["current_stage"] == "ROAD_FUSION"
    assert "Hybrid road fusion failed" in data["error"]
    assert "RuntimeError" in data["technical_error"]


# ─────────────────────────────────────────────────────────────────────────────
# 7. Analysis Summary Test
# ─────────────────────────────────────────────────────────────────────────────
def test_analysis_summary():
    """Verify GET /api/results/{id}/summary returns complete metrics."""
    resp = client.get(f"/api/results/{DEMO_ID}/summary")
    assert resp.status_code == 200
    summary = resp.json()

    assert "analysis_id" in summary
    assert "status" in summary
    assert "building_count" in summary
    assert summary["building_count"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Output Isolation Test
# ─────────────────────────────────────────────────────────────────────────────
def test_output_directory_isolation():
    """Verify new analysis creates its own directory and preserves demo analysis."""
    new_id = f"iso_{uuid.uuid4().hex[:8]}"
    new_dir = storage.create_analysis_dir(new_id)

    assert new_dir.is_dir()
    assert new_id in str(new_dir)

    # Ensure demo directory is completely untouched
    demo_dir = storage.get_analysis_dir(DEMO_ID)
    assert demo_dir.is_dir()
    assert (demo_dir / "buildings.geojson").is_file()

    # Clean up test dir
    import shutil
    shutil.rmtree(new_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Deliverables Export Endpoints Test
# ─────────────────────────────────────────────────────────────────────────────
def test_export_endpoints():
    """Verify all export deliverables endpoints return valid content."""
    # Reviewed Parcels GeoJSON
    resp_rev = client.get(f"/api/results/{DEMO_ID}/parcel-reviews/export")
    assert resp_rev.status_code == 200
    rev_data = resp_rev.json()
    assert rev_data["type"] == "FeatureCollection"
    assert len(rev_data["features"]) == 254

    # Audit Trail JSON
    resp_audit = client.get(f"/api/results/{DEMO_ID}/parcel-reviews/audit")
    assert resp_audit.status_code == 200
    assert isinstance(resp_audit.json(), list)

    # Full Summary JSON
    resp_sum = client.get(f"/api/results/{DEMO_ID}/summary")
    assert resp_sum.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# 10. Existing Parcel Review Compatibility Test
# ─────────────────────────────────────────────────────────────────────────────
def test_parcel_review_compatibility():
    """Verify Phase 14B review queue and candidate decision APIs function cleanly."""
    # Fetch review list
    resp = client.get(f"/api/results/{DEMO_ID}/parcel-reviews")
    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"]["total"] == 254

    # Fetch candidate P56
    p56_resp = client.get(f"/api/results/{DEMO_ID}/parcel-reviews/56")
    assert p56_resp.status_code == 200
    p56_data = p56_resp.json()
    assert p56_data["candidate_id"] == 56
    assert "confidence" in p56_data
    assert "confidence_band" in p56_data
