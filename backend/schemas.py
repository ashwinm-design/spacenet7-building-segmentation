"""
backend/schemas.py

Pydantic models for request validation and response serialization.
"""

from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Service status", examples=["ok"])
    service: str = Field(..., description="Service name", examples=["SIH26012 AI Mapping Backend"])


class AnalysisResponse(BaseModel):
    """Result summary returned by the POST /api/analyze and GET /api/results/{analysis_id} endpoints."""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    status: str = Field(..., description="Execution status ('completed', 'failed')", examples=["completed"])
    building_count: int = Field(..., description="Total number of building instances extracted")
    total_area: float = Field(..., description="Total built-up footprint area")
    area_unit: str = Field(..., description="Unit of measurement ('m^2' or 'pixels^2')", examples=["m^2"])
    average_building_area: float = Field(..., description="Mean building footprint area")
    average_confidence: float = Field(..., description="Mean footprint classification confidence [0.0 - 1.0]")
    processing_time: float = Field(..., description="Total processing time in seconds")
    crs: Optional[str] = Field(None, description="Coordinate Reference System if georeferenced", examples=["EPSG:32732"])
    georeferenced: bool = Field(..., description="True if input has geospatial reference")
    geojson_url: str = Field(..., description="API endpoint to download the GeoJSON")
    statistics_url: str = Field(..., description="API endpoint to download the statistics CSV")
    overlay_url: str = Field(..., description="API endpoint to download the PNG overlay")


class BuildingDetailResponse(BaseModel):
    """Detail of a single extracted building footprint."""
    id: int = Field(..., description="Sequential building identifier")
    area: float = Field(..., description="Footprint area")
    perimeter: float = Field(..., description="Boundary perimeter")
    area_unit: str = Field(..., description="Units for area ('m^2' or 'pixels^2')")
    perimeter_unit: str = Field(..., description="Units for perimeter ('meters' or 'pixels')")
    centroid_x: float = Field(..., description="Centroid X (Easting or Pixel X)")
    centroid_y: float = Field(..., description="Centroid Y (Northing or Pixel Y)")
    confidence: float = Field(..., description="Mean classification confidence")
    min_confidence: float = Field(..., description="Minimum classification confidence")
    is_georeferenced: bool = Field(..., description="True if georeferenced")
    geometry: Optional[Dict[str, Any]] = Field(None, description="GeoJSON geometry object")
    feature: Optional[Dict[str, Any]] = Field(None, description="Full GeoJSON Feature object")


class ErrorResponse(BaseModel):
    """Structured error response schema."""
    detail: str = Field(..., description="Human-readable error explanation")
    error_code: Optional[str] = Field(None, description="Machine-readable error classification")


# ════════════════════════════════════════════════════════════════════
# Parcel Review Schemas (Phase 14B)
# ════════════════════════════════════════════════════════════════════
class ParcelReviewItem(BaseModel):
    """Review state for a preliminary parcel candidate."""
    candidate_id: int = Field(..., description="Unique parcel candidate identifier")
    analysis_id: str = Field(..., description="Analysis identifier")
    review_status: str = Field(
        ...,
        description="Current review status ('unreviewed', 'accepted', 'edited', 'rejected', 'needs_review')",
        examples=["unreviewed"]
    )
    reviewer_note: Optional[str] = Field(None, description="Human reviewer notes or observations")
    confidence: float = Field(..., description="AI confidence score [0.0 - 1.0]")
    confidence_band: str = Field(..., description="Confidence tier ('HIGH', 'MODERATE', 'LOW', 'VERY_LOW')")
    flags: Union[List[str], str] = Field(default="", description="Uncertainty flags (list or comma-separated)")
    original_geometry: Dict[str, Any] = Field(..., description="Original AI-generated GeoJSON geometry")
    current_geometry: Dict[str, Any] = Field(..., description="Current GeoJSON geometry (human-edited or original)")
    geometry_modified: bool = Field(False, description="True if geometry was edited by reviewer")
    review_timestamp: Optional[str] = Field(None, description="ISO-8601 UTC timestamp of last review action")

    # Geometry-dependent and explainability metrics for inspection panel
    area_m2: Optional[float] = Field(None, description="Current parcel area in m²")
    building_coverage: Optional[float] = Field(None, description="Footprint area to parcel area ratio")
    compactness: Optional[float] = Field(None, description="Polsby-Popper compactness metric [0.0 - 1.0]")
    road_frontage: Optional[float] = Field(None, description="Road corridor frontage length in meters")
    nearest_road_distance: Optional[float] = Field(None, description="Distance to nearest physical road corridor")
    conf_building_evidence: Optional[float] = Field(None, description="Building evidence score")
    conf_road_access: Optional[float] = Field(None, description="Road access evidence score")
    conf_geometry_quality: Optional[float] = Field(None, description="Geometry quality score")
    conf_topological_plausibility: Optional[float] = Field(None, description="Topological plausibility score")
    recommendation: Optional[str] = Field(None, description="System recommendation narrative")


class ParcelReviewUpdateRequest(BaseModel):
    """Payload to accept, flag, edit, or reject a parcel candidate."""
    review_status: str = Field(
        ...,
        description="Review status: 'unreviewed', 'accepted', 'edited', 'rejected', 'needs_review'"
    )
    reviewer_note: Optional[str] = Field(None, description="Reviewer notes")
    geometry: Optional[Dict[str, Any]] = Field(None, description="Optional edited GeoJSON geometry")
    current_geometry: Optional[Dict[str, Any]] = Field(None, description="Alternative field for edited geometry")


class ParcelReviewSummary(BaseModel):
    """Aggregate counts for the parcel review queue."""
    total: int = Field(..., description="Total parcel candidates")
    unreviewed: int = Field(0, description="Candidates awaiting review")
    accepted: int = Field(0, description="Candidates accepted without edits")
    edited: int = Field(0, description="Candidates with human-edited geometry")
    needs_review: int = Field(0, description="Candidates flagged for closer inspection")
    rejected: int = Field(0, description="Candidates rejected from cadastral hypothesis")


class ParcelReviewListResponse(BaseModel):
    """Summary and list of parcel reviews."""
    analysis_id: str = Field(..., description="Analysis identifier")
    summary: ParcelReviewSummary = Field(..., description="Review status breakdown counts")
    reviews: List[ParcelReviewItem] = Field(..., description="List of parcel review states")


# ════════════════════════════════════════════════════════════════════
# Pipeline Orchestration Schemas (Phase 15)
# ════════════════════════════════════════════════════════════════════
class AnalysisCreationResponse(BaseModel):
    """Initial response when an analysis job is created/queued."""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    status: str = Field("queued", description="Initial queue state ('queued')")
    current_stage: str = Field("QUEUED", description="Initial stage name")
    progress: int = Field(0, description="Initial stage progress [0 - 100]")
    message: Optional[str] = Field(None, description="Optional informational message")


class PipelineStatusResponse(BaseModel):
    """Real-time pipeline execution progress schema."""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    status: str = Field(..., description="Execution status ('queued', 'running', 'completed', 'failed')")
    progress: int = Field(..., description="Weighted stage completion percentage [0 - 100]")
    current_stage: str = Field(..., description="Currently active pipeline stage")
    completed_stages: List[str] = Field(default_factory=list, description="Stages completed so far")
    error: Optional[str] = Field(None, description="Human-readable error description if failed")
    technical_error: Optional[str] = Field(None, description="Technical error identifier or exception name")


class AnalysisSummaryData(BaseModel):
    """Comprehensive analysis summary schema for analysis_summary.json."""
    analysis_id: str = Field(..., description="Analysis identifier")
    status: str = Field(..., description="Completion status ('completed', 'failed')")
    input_dimensions: List[int] = Field(..., description="Image dimensions [height, width]")
    crs: Optional[str] = Field(None, description="Coordinate Reference System if georeferenced")
    georeferenced: bool = Field(..., description="True if georeferenced")
    processing_time: float = Field(..., description="Total wall-clock runtime in seconds")
    stage_timings: Dict[str, float] = Field(..., description="Measured runtime per pipeline stage in seconds")
    building_count: int = Field(..., description="Total building footprints detected")
    road_corridor_count: int = Field(..., description="Total hybrid road corridors extracted")
    road_length_m: float = Field(..., description="Total centerline length in meters")
    parcel_candidate_count: int = Field(..., description="Total preliminary parcel candidates generated")
    geometry_validity_pct: float = Field(..., description="Percentage of valid polygon geometries [0.0 - 100.0]")
    confidence_statistics: Dict[str, float] = Field(..., description="Mean, median, min, max confidence")
    flagged_parcel_count: int = Field(..., description="Number of candidates with uncertainty flags")
    review_status_counts: Dict[str, int] = Field(..., description="Breakdown of human review statuses")
    disclaimer: str = Field(
        default="AI-generated parcel boundaries are preliminary spatial candidates and require professional cadastral/survey verification. Review status does not establish legal ownership or boundary rights.",
        description="Scientific and legal disclaimer"
    )


