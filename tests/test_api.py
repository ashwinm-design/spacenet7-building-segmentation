"""
tests/test_api.py

Automated tests for SIH26012 FastAPI backend endpoints using pytest and TestClient.
Tests:
  1. GET /api/health
  2. Invalid file extension rejection (400)
  3. Empty upload rejection (400)
  4. Small RGB image analysis (POST /api/analyze)
  5. Open Cities GeoTIFF ROI analysis (POST /api/analyze with ROI)
  6. Result summary retrieval (GET /api/results/{analysis_id})
  7. GeoJSON download (GET /api/results/{analysis_id}/geojson)
  8. Statistics CSV download (GET /api/results/{analysis_id}/statistics)
  9. Building detail endpoint (GET /api/results/{analysis_id}/buildings/{id})
 10. Missing analysis ID / invalid building ID handling (404)
"""

import os
import sys
import io
import json
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.main import app

client = TestClient(app)

# Test file paths
SAMPLE_RGB_IMAGE = "results/experiments/open_cities_zero_shot/images/f49f31_r3072_c5120.png"
SAMPLE_GEOTIFF = "data/open_cities_raw/ptn/f49f31/f49f31.tif"


# 1. Health check endpoint
def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "SIH26012" in data["service"]


# 2. Invalid file type rejection
def test_invalid_file_type():
    file_bytes = b"dummy text content"
    response = client.post(
        "/api/analyze",
        files={"file": ("malicious_script.sh", file_bytes, "application/x-sh")},
    )
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


# 3. Empty file upload rejection
def test_empty_file_upload():
    empty_bytes = b""
    response = client.post(
        "/api/analyze",
        files={"file": ("empty.png", empty_bytes, "image/png")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


# 4. Analyze existing small RGB image
def test_analyze_rgb_image():
    if not os.path.exists(SAMPLE_RGB_IMAGE):
        pytest.skip(f"Sample RGB image not found: {SAMPLE_RGB_IMAGE}")

    with open(SAMPLE_RGB_IMAGE, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/analyze",
        files={"file": ("sample_patch.png", file_bytes, "image/png")},
    )
    assert response.status_code == 201
    data = response.json()
    assert "analysis_id" in data
    assert data["status"] == "completed"
    assert data["building_count"] > 0
    assert data["georeferenced"] is False
    assert data["area_unit"] == "pixels^2"
    assert "geojson_url" in data
    assert "statistics_url" in data
    assert "overlay_url" in data

    # Verify result retrieval for this analysis
    analysis_id = data["analysis_id"]

    # Test summary retrieval
    sum_resp = client.get(f"/api/results/{analysis_id}")
    assert sum_resp.status_code == 200
    assert sum_resp.json()["analysis_id"] == analysis_id

    # Test GeoJSON retrieval
    geo_resp = client.get(f"/api/results/{analysis_id}/geojson")
    assert geo_resp.status_code == 200
    gj = geo_resp.json()
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == data["building_count"]

    # Test statistics retrieval
    stat_resp = client.get(f"/api/results/{analysis_id}/statistics")
    assert stat_resp.status_code == 200
    assert "id,area,perimeter" in stat_resp.text

    # Test overlay retrieval
    overlay_resp = client.get(f"/api/results/{analysis_id}/overlay")
    assert overlay_resp.status_code == 200
    assert overlay_resp.headers["content-type"] == "image/png"

    # Test building detail retrieval (building ID = 1)
    bldg_resp = client.get(f"/api/results/{analysis_id}/buildings/1")
    assert bldg_resp.status_code == 200
    bldg_data = bldg_resp.json()
    assert bldg_data["id"] == 1
    assert "area" in bldg_data
    assert "confidence" in bldg_data
    assert "geometry" in bldg_data


# 5. Analyze existing Open Cities GeoTIFF ROI
def test_analyze_geotiff_roi():
    if not os.path.exists(SAMPLE_GEOTIFF):
        pytest.skip(f"Sample GeoTIFF not found: {SAMPLE_GEOTIFF}")

    with open(SAMPLE_GEOTIFF, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/analyze",
        files={"file": ("f49f31_sample.tif", file_bytes, "image/tiff")},
        data={"roi": "3072,2048,4096,3072"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "completed"
    assert data["georeferenced"] is True
    assert "EPSG:32732" in data["crs"]
    assert data["area_unit"] == "m^2"
    assert data["building_count"] == 254
    assert data["total_area"] > 10000.0

    analysis_id = data["analysis_id"]

    # Check building detail
    bldg_resp = client.get(f"/api/results/{analysis_id}/buildings/254")
    assert bldg_resp.status_code == 200
    bldg_data = bldg_resp.json()
    assert bldg_data["id"] == 254
    assert bldg_data["is_georeferenced"] is True


# 6. Missing analysis ID handling
def test_nonexistent_analysis_id():
    fake_id = "nonexistent_9999"
    assert client.get(f"/api/results/{fake_id}").status_code == 404
    assert client.get(f"/api/results/{fake_id}/geojson").status_code == 404
    assert client.get(f"/api/results/{fake_id}/statistics").status_code == 404
    assert client.get(f"/api/results/{fake_id}/overlay").status_code == 404
    assert client.get(f"/api/results/{fake_id}/buildings/1").status_code == 404
