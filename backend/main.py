"""
backend/main.py

FastAPI Application for SIH26012 Building & Cadastral Feature Extraction Backend.
Exposes REST API endpoints for image upload, tiled ML inference, and result retrieval.
"""

import os
import re
import json
import uuid
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    HealthResponse,
    AnalysisResponse,
    BuildingDetailResponse,
    ErrorResponse,
    ParcelReviewItem,
    ParcelReviewUpdateRequest,
    ParcelReviewSummary,
    ParcelReviewListResponse,
    AnalysisCreationResponse,
    PipelineStatusResponse,
    AnalysisSummaryData,
)
from backend.storage import storage
from backend.storage.reviews import review_storage
from backend.services.inference_service import run_analysis_pipeline
from backend.services.pipeline_service import (
    run_analysis,
    run_analysis_async,
    get_pipeline_status,
    validate_input_imagery,
)
from backend.services.result_service import (
    get_analysis_metadata,
    get_building_feature,
)

# ════════════════════════════════════════════════════════════════════
# FastAPI Application & CORS Configuration
# ════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="SIH26012 AI Mapping Backend",
    description=(
        "High-resolution aerial and drone building-footprint segmentation "
        "and cadastral feature extraction REST API."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allowed CORS origins for local development (React/Vite)
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
MAX_UPLOAD_SIZE = 250 * 1024 * 1024  # 250 MB


def parse_roi(roi_str: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    """Parses ROI string formatted as 'xmin,ymin,xmax,ymax' or 'xmin ymin xmax ymax'."""
    if not roi_str or not roi_str.strip():
        return None
    # Split by comma or whitespace
    parts = re.split(r"[,\s]+", roi_str.strip())
    if len(parts) != 4:
        raise ValueError("ROI must contain exactly 4 integers: xmin ymin xmax ymax")
    coords = tuple(int(p) for p in parts)
    if coords[2] <= coords[0] or coords[3] <= coords[1]:
        raise ValueError("ROI invalid: xmax must be > xmin and ymax must be > ymin")
    return coords


# ════════════════════════════════════════════════════════════════════
# 1. Health Endpoint
# ════════════════════════════════════════════════════════════════════
@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
def health_check() -> HealthResponse:
    """Returns the operational status of the AI mapping service."""
    return HealthResponse(
        status="ok",
        service="SIH26012 AI Mapping Backend"
    )


# ════════════════════════════════════════════════════════════════════
# 2. Analysis Execution Endpoint
# ════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════
# 2. Analysis Execution Endpoints
# ════════════════════════════════════════════════════════════════════
@app.post(
    "/api/analyze",
    response_model=Union[AnalysisCreationResponse, AnalysisResponse],
    status_code=status.HTTP_201_CREATED,
    tags=["Analysis"],
    summary="Analyze aerial or drone image",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format, empty content, or corrupted raster"},
        413: {"model": ErrorResponse, "description": "File size exceeds 250MB limit"},
        500: {"model": ErrorResponse, "description": "Inference pipeline error"},
    },
)
async def analyze_image(
    file: UploadFile = File(..., description="Aerial or drone imagery file (GeoTIFF, PNG, JPG)"),
    roi: Optional[str] = Form(None, description="Optional bounding box: 'xmin,ymin,xmax,ymax'"),
    background: bool = Form(False, description="Run in background task if True, synchronous if False"),
) -> Union[AnalysisCreationResponse, AnalysisResponse]:
    """
    Uploads an aerial image, validates raster integrity, and executes the complete SIH26012 pipeline:
    Building segmentation, classical + DL road extraction, hybrid fusion, preliminary parcel generation,
    parcel intelligence, and verification review queue initialization.
    """
    # 1. Validate file extension
    orig_filename = file.filename or "upload.png"
    ext = Path(orig_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # 2. Read and validate file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({len(content)/(1024*1024):.1f} MB) exceeds maximum allowed limit of 250 MB.",
        )

    # 3. Parse optional ROI
    parsed_roi = None
    if roi:
        try:
            parsed_roi = parse_roi(roi)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid ROI parameter: {str(e)}",
            )

    # 4. Generate unique analysis ID and save upload
    analysis_id = uuid.uuid4().hex[:10]
    saved_file_path = storage.save_uploaded_file(analysis_id, orig_filename, content)

    # Validate raster integrity (dimensions, channels, NaN/Inf)
    try:
        validate_input_imagery(saved_file_path)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input raster validation failed: {str(e)}",
        )

    # 5. Background vs Synchronous execution
    if background:
        run_analysis_async(analysis_id, saved_file_path, roi=parsed_roi)
        return AnalysisCreationResponse(
            analysis_id=analysis_id,
            status="queued",
            current_stage="QUEUED",
            progress=0,
            message="Analysis pipeline queued in background.",
        )

    # Synchronous execution
    try:
        run_analysis(analysis_id, saved_file_path, roi=parsed_roi)
        metadata = storage.get_metadata(analysis_id)
        if not metadata:
            raise RuntimeError("Pipeline completed but metadata.json was not generated.")
        return AnalysisResponse(**metadata)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(e)}",
        )


@app.post(
    "/api/demo",
    response_model=Union[AnalysisCreationResponse, AnalysisResponse],
    status_code=status.HTTP_201_CREATED,
    tags=["Analysis"],
    summary="Run inference on default Open Cities demo sample (ROI 3072,2048,4096,3072)",
)
def run_demo_analysis(
    background: bool = False,
) -> Union[AnalysisCreationResponse, AnalysisResponse]:
    """Runs automated end-to-end pipeline on the locked Open Cities GeoTIFF subregion."""
    demo_image = Path("data/open_cities_raw/ptn/f49f31/f49f31.tif")
    if not demo_image.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Demo image not found at {demo_image}",
        )
    analysis_id = uuid.uuid4().hex[:10]
    with open(demo_image, "rb") as f:
        content = f.read()
    saved_file_path = storage.save_uploaded_file(analysis_id, "f49f31_demo.tif", content)

    if background:
        run_analysis_async(analysis_id, saved_file_path, roi=(3072, 2048, 4096, 3072))
        return AnalysisCreationResponse(
            analysis_id=analysis_id,
            status="queued",
            current_stage="QUEUED",
            progress=0,
            message="Demo analysis pipeline queued in background.",
        )

    try:
        run_analysis(analysis_id, saved_file_path, roi=(3072, 2048, 4096, 3072))
        metadata = storage.get_metadata(analysis_id)
        if not metadata:
            raise RuntimeError("Demo pipeline completed but metadata.json was not generated.")
        return AnalysisResponse(**metadata)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(e)}",
        )


@app.get(
    "/api/analysis/{analysis_id}/status",
    response_model=PipelineStatusResponse,
    tags=["Analysis"],
    summary="Get real-time pipeline execution progress status",
)
def get_analysis_status_endpoint(analysis_id: str) -> PipelineStatusResponse:
    """Returns stage-weighted progress, active stage, completed stages, and error information."""
    status_data = get_pipeline_status(analysis_id)
    if status_data.get("status") == "not_found":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis ID '{analysis_id}' not found.",
        )
    return PipelineStatusResponse(**status_data)


@app.get(
    "/api/results/{analysis_id}/summary",
    tags=["Results"],
    summary="Download complete analysis summary JSON",
)
def get_analysis_full_summary(analysis_id: str):
    """Returns the comprehensive analysis_summary.json containing KPI metrics and timings."""
    summary_path = storage.get_analysis_dir(analysis_id) / "analysis_summary.json"
    if not summary_path.is_file():
        legacy_meta = storage.get_metadata(analysis_id)
        if legacy_meta:
            return JSONResponse(content=legacy_meta)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis summary for analysis ID '{analysis_id}' not found.",
        )
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)


@app.get(
    "/api/results/{analysis_id}/parcel-reviews/audit",
    tags=["Parcel Verification"],
    summary="Download parcel review audit trail JSON",
)
def get_parcel_reviews_audit_log(analysis_id: str):
    """Returns the complete immutable review change audit log."""
    audit_path = review_storage.get_audit_path(analysis_id)
    if not audit_path.is_file():
        return JSONResponse(content=[])
    with open(audit_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)



# ════════════════════════════════════════════════════════════════════
# 3. Result Retrieval Endpoints
# ════════════════════════════════════════════════════════════════════
@app.get(
    "/api/results/{analysis_id}",
    response_model=AnalysisResponse,
    tags=["Results"],
    summary="Get analysis summary metadata",
    responses={
        404: {"model": ErrorResponse, "description": "Analysis ID not found"},
    },
)
def get_analysis_summary(analysis_id: str) -> AnalysisResponse:
    """Returns summary metadata and KPI statistics for a completed analysis."""
    meta = get_analysis_metadata(analysis_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis ID '{analysis_id}' not found.",
        )
    return AnalysisResponse(**meta)


@app.get(
    "/api/results/{analysis_id}/geojson",
    tags=["Results"],
    summary="Download extracted GeoJSON building footprints",
    responses={
        200: {"content": {"application/geo+json": {}}, "description": "GeoJSON FeatureCollection"},
        404: {"model": ErrorResponse, "description": "GeoJSON file not found"},
    },
)
def get_geojson(analysis_id: str):
    """Returns the generated GeoJSON file preserving source CRS coordinates."""
    path = storage.get_geojson_path(analysis_id)
    if not path or not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"GeoJSON for analysis ID '{analysis_id}' not found.",
        )
    return FileResponse(
        path=str(path),
        media_type="application/geo+json",
        filename=f"buildings_{analysis_id}.geojson",
    )


@app.get(
    "/api/results/{analysis_id}/roads",
    tags=["Results"],
    summary="Download extracted physical road and access corridors GeoJSON",
    responses={
        200: {"content": {"application/geo+json": {}}, "description": "GeoJSON FeatureCollection of physical road corridors"},
        404: {"model": ErrorResponse, "description": "Road corridors layer not found"},
    },
)
def get_roads_geojson(analysis_id: str, source: str = "classical"):
    """Returns the physical road corridors GeoJSON layer in source CRS coordinates (source: classical, deep_learning, or hybrid)."""
    path = storage.get_roads_geojson_path(analysis_id, source=source)
    if not path or not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Road corridors GeoJSON for analysis ID '{analysis_id}' (source='{source}') not found.",
        )
    return FileResponse(
        path=str(path),
        media_type="application/geo+json",
        filename=f"roads_{source}_{analysis_id}.geojson",
    )


@app.get(
    "/api/results/{analysis_id}/parcels",
    tags=["Results"],
    summary="Download preliminary parcel candidates GeoJSON",
    responses={
        200: {"content": {"application/geo+json": {}}, "description": "GeoJSON FeatureCollection of preliminary parcel candidates"},
        404: {"model": ErrorResponse, "description": "Parcel candidates layer not found"},
    },
)
def get_parcels_geojson(analysis_id: str, source: str = "classical"):
    """Returns the road-constrained preliminary parcel candidates GeoJSON layer in source CRS coordinates (source: classical, deep_learning, or hybrid)."""
    if source in ("hybrid", "reviewed"):
        reviewed_p = review_storage.get_reviewed_geojson_path(analysis_id)
        if not reviewed_p.is_file():
            try:
                review_storage.export_reviewed_geojson(analysis_id)
            except Exception:
                pass
        if reviewed_p.is_file():
            return FileResponse(
                path=str(reviewed_p),
                media_type="application/geo+json",
                filename=f"parcels_{source}_{analysis_id}.geojson",
            )

    path = storage.get_parcels_geojson_path(analysis_id, source=source)
    if not path or not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preliminary parcel candidates GeoJSON for analysis ID '{analysis_id}' (source='{source}') not found.",
        )
    return FileResponse(
        path=str(path),
        media_type="application/geo+json",
        filename=f"parcels_{source}_{analysis_id}.geojson",
    )


@app.get(
    "/api/results/{analysis_id}/statistics",
    tags=["Results"],
    summary="Download building attributes statistics CSV",
    responses={
        200: {"content": {"text/csv": {}}, "description": "CSV table with instance dimensions"},
        404: {"model": ErrorResponse, "description": "Statistics CSV not found"},
    },
)
def get_statistics(analysis_id: str):
    """Returns the instance-level building footprint dimensions and metrics in CSV format."""
    path = storage.get_statistics_path(analysis_id)
    if not path or not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Statistics CSV for analysis ID '{analysis_id}' not found.",
        )
    return FileResponse(
        path=str(path),
        media_type="text/csv",
        filename=f"statistics_{analysis_id}.csv",
    )


@app.get(
    "/api/results/{analysis_id}/overlay",
    tags=["Results"],
    summary="Download PNG visual overlay",
    responses={
        200: {"content": {"image/png": {}}, "description": "High-resolution overlay PNG"},
        404: {"model": ErrorResponse, "description": "Overlay image not found"},
    },
)
def get_overlay_image(analysis_id: str):
    """Returns the aesthetic PNG overlay image with vector footprints drawn over the input."""
    path = storage.get_overlay_path(analysis_id)
    if not path or not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Overlay image for analysis ID '{analysis_id}' not found.",
        )
    return FileResponse(
        path=str(path),
        media_type="image/png",
        filename=f"overlay_{analysis_id}.png",
    )


# ════════════════════════════════════════════════════════════════════
# 4. Building Detail Endpoint
# ════════════════════════════════════════════════════════════════════
@app.get(
    "/api/results/{analysis_id}/buildings/{building_id}",
    response_model=BuildingDetailResponse,
    tags=["Results"],
    summary="Get single building footprint detail by ID",
    responses={
        404: {"model": ErrorResponse, "description": "Analysis ID or Building ID not found"},
    },
)
def get_building_detail(analysis_id: str, building_id: int) -> BuildingDetailResponse:
    """Returns detailed geometric properties and GeoJSON geometry for a single building instance."""
    if not storage.exists(analysis_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis ID '{analysis_id}' not found.",
        )

    bldg = get_building_feature(analysis_id, building_id)
    if not bldg:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Building #{building_id} not found in analysis '{analysis_id}'.",
        )

    return BuildingDetailResponse(**bldg)


# ════════════════════════════════════════════════════════════════════
# 5. Human-in-the-Loop Parcel Review Endpoints (Phase 14B)
# ════════════════════════════════════════════════════════════════════
@app.get(
    "/api/results/{analysis_id}/parcel-reviews",
    response_model=ParcelReviewListResponse,
    tags=["Parcel Verification"],
    summary="Get all parcel reviews and queue summary",
    responses={
        404: {"model": ErrorResponse, "description": "Analysis ID not found"},
    },
)
def get_parcel_reviews(analysis_id: str) -> ParcelReviewListResponse:
    """Returns review state and uncertainty metrics for all preliminary parcel candidates."""
    if not storage.exists(analysis_id) and not review_storage.get_reviews_path(analysis_id).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis ID '{analysis_id}' not found.",
        )

    reviews_dict = review_storage.get_all_reviews(analysis_id)
    summary_dict = review_storage.get_summary(analysis_id)

    reviews_list = [ParcelReviewItem(**item) for item in sorted(reviews_dict.values(), key=lambda x: int(x["candidate_id"]))]
    summary_obj = ParcelReviewSummary(**summary_dict)

    return ParcelReviewListResponse(
        analysis_id=analysis_id,
        summary=summary_obj,
        reviews=reviews_list,
    )


@app.get(
    "/api/results/{analysis_id}/parcel-reviews/summary",
    response_model=ParcelReviewSummary,
    tags=["Parcel Verification"],
    summary="Get review queue breakdown summary",
    responses={
        404: {"model": ErrorResponse, "description": "Analysis ID not found"},
    },
)
def get_parcel_review_summary(analysis_id: str) -> ParcelReviewSummary:
    """Returns aggregate counts (total, unreviewed, accepted, edited, needs_review, rejected)."""
    if not storage.exists(analysis_id) and not review_storage.get_reviews_path(analysis_id).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis ID '{analysis_id}' not found.",
        )

    summary = review_storage.get_summary(analysis_id)
    return ParcelReviewSummary(**summary)


@app.get(
    "/api/results/{analysis_id}/parcel-reviews/export",
    tags=["Parcel Verification"],
    summary="Export GIS-ready reviewed parcels GeoJSON",
    responses={
        200: {"content": {"application/geo+json": {}}, "description": "GeoJSON FeatureCollection with review audit metadata"},
        404: {"model": ErrorResponse, "description": "Analysis ID not found"},
    },
)
def export_parcel_reviews_geojson(analysis_id: str):
    """Exports all reviewed parcel candidates as GIS-ready GeoJSON with complete audit trail attributes."""
    if not storage.exists(analysis_id) and not review_storage.get_reviews_path(analysis_id).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis ID '{analysis_id}' not found.",
        )

    try:
        review_storage.export_reviewed_geojson(analysis_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export reviewed parcel candidates: {str(e)}",
        )

    export_path = review_storage.get_reviewed_geojson_path(analysis_id)
    if not export_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Reviewed GeoJSON file could not be generated.",
        )

    return FileResponse(
        path=str(export_path),
        media_type="application/geo+json",
        filename=f"reviewed_parcels_{analysis_id}.geojson",
    )


@app.get(
    "/api/results/{analysis_id}/parcel-reviews/{candidate_id}",
    response_model=ParcelReviewItem,
    tags=["Parcel Verification"],
    summary="Get single parcel candidate review state",
    responses={
        404: {"model": ErrorResponse, "description": "Analysis ID or Candidate ID not found"},
    },
)
def get_parcel_review_item(analysis_id: str, candidate_id: int) -> ParcelReviewItem:
    """Returns review status, evidence metrics, confidence band, and geometry for a single candidate."""
    if not storage.exists(analysis_id) and not review_storage.get_reviews_path(analysis_id).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis ID '{analysis_id}' not found.",
        )

    rev = review_storage.get_review(analysis_id, candidate_id)
    if not rev:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Parcel candidate #{candidate_id} not found in analysis '{analysis_id}'.",
        )

    return ParcelReviewItem(**rev)


@app.post(
    "/api/results/{analysis_id}/parcel-reviews/{candidate_id}",
    response_model=ParcelReviewItem,
    tags=["Parcel Verification"],
    summary="Update parcel candidate review state (accept, flag, edit, reject)",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid review parameters or invalid geometry"},
        404: {"model": ErrorResponse, "description": "Analysis ID or Candidate ID not found"},
    },
)
def update_parcel_review_post(
    analysis_id: str,
    candidate_id: int,
    payload: ParcelReviewUpdateRequest,
) -> ParcelReviewItem:
    """Accepts, flags for review, edits geometry, or rejects a preliminary parcel candidate."""
    if not storage.exists(analysis_id) and not review_storage.get_reviews_path(analysis_id).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis ID '{analysis_id}' not found.",
        )

    # Geometry can be supplied via `geometry` or `current_geometry`
    new_geom = payload.geometry or payload.current_geometry

    try:
        updated = review_storage.update_review(
            analysis_id=analysis_id,
            candidate_id=candidate_id,
            review_status=payload.review_status,
            reviewer_note=payload.reviewer_note,
            new_geometry=new_geom,
        )
        # Keep export synced
        review_storage.export_reviewed_geojson(analysis_id)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e).strip("'"),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving parcel review: {str(e)}",
        )

    return ParcelReviewItem(**updated)


@app.patch(
    "/api/results/{analysis_id}/parcel-reviews/{candidate_id}",
    response_model=ParcelReviewItem,
    tags=["Parcel Verification"],
    summary="Patch parcel candidate review state (RESTful alternative to POST)",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid review parameters or invalid geometry"},
        404: {"model": ErrorResponse, "description": "Analysis ID or Candidate ID not found"},
    },
)
def update_parcel_review_patch(
    analysis_id: str,
    candidate_id: int,
    payload: ParcelReviewUpdateRequest,
) -> ParcelReviewItem:
    """PATCH endpoint for parcel review updates."""
    return update_parcel_review_post(analysis_id, candidate_id, payload)


@app.post(
    "/api/results/{analysis_id}/parcel-reviews/{candidate_id}/reset",
    response_model=ParcelReviewItem,
    tags=["Parcel Verification"],
    summary="Reset parcel candidate geometry back to original AI hypothesis",
    responses={
        404: {"model": ErrorResponse, "description": "Analysis ID or Candidate ID not found"},
    },
)
def reset_parcel_geometry(analysis_id: str, candidate_id: int) -> ParcelReviewItem:
    """Restores original AI-generated parcel boundary, reverts metrics, and logs reset in audit trail."""
    if not storage.exists(analysis_id) and not review_storage.get_reviews_path(analysis_id).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis ID '{analysis_id}' not found.",
        )

    try:
        updated = review_storage.reset_review_geometry(analysis_id, candidate_id)
        review_storage.export_reviewed_geojson(analysis_id)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e).strip("'"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error resetting parcel geometry: {str(e)}",
        )

    return ParcelReviewItem(**updated)

