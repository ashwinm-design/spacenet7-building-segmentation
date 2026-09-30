/**
 * frontend/src/types/analysis.ts
 *
 * TypeScript interface definitions for SIH26012 GIS AI Mapping Backend.
 */

export interface HealthResponse {
  status: string;
  service: string;
}

export interface AnalysisResponse {
  analysis_id: string;
  status: 'completed' | 'failed' | string;
  building_count: number;
  total_area: number;
  area_unit: string;
  average_building_area: number;
  average_confidence: number;
  processing_time: number;
  crs: string | null;
  georeferenced: boolean;
  geojson_url: string;
  statistics_url: string;
  overlay_url: string;
}

export interface BuildingProperties {
  id: number;
  area: number;
  perimeter: number;
  area_unit?: string;
  perimeter_unit?: string;
  confidence: number;
  min_confidence?: number;
  centroid_x?: number;
  centroid_y?: number;
}

export interface ParcelProperties {
  parcel_id: number;
  candidate_id?: number;
  building_id: number;
  cluster_id: number;
  method: string;
  parcel_area: number;
  parcel_perimeter: number;
  building_area: number;
  coverage_ratio: number;
  containment_score: number;
  compactness: number;
  road_touches?: boolean;
  road_distance_m?: number | null;
  road_frontage_m?: number;
  road_evidence_score?: number | null;
  candidate_score: number;
  road_evidence: string;
  topology_status: string;

  // Phase 14A & 14B attributes
  review_status?: ReviewStatus;
  confidence?: number;
  confidence_band?: ConfidenceBand;
  flags?: string | string[];
  reviewer_note?: string | null;
  geometry_modified?: boolean;
  review_timestamp?: string | null;
  area_m2?: number;
  building_coverage?: number;
  road_frontage?: number;
  nearest_road_distance?: number;
  conf_building_evidence?: number;
  conf_road_access?: number;
  conf_geometry_quality?: number;
  conf_topological_plausibility?: number;
  recommendation?: string;
}

export type ReviewStatus = 'unreviewed' | 'accepted' | 'edited' | 'rejected' | 'needs_review';
export type ConfidenceBand = 'HIGH' | 'MODERATE' | 'LOW' | 'VERY_LOW';

export interface ParcelReviewItem {
  candidate_id: number;
  analysis_id: string;
  review_status: ReviewStatus;
  reviewer_note: string | null;
  confidence: number;
  confidence_band: ConfidenceBand;
  flags: string | string[];
  original_geometry: GeoJSONGeometry;
  current_geometry: GeoJSONGeometry;
  geometry_modified: boolean;
  review_timestamp: string | null;
  area_m2?: number;
  building_coverage?: number;
  compactness?: number;
  road_frontage?: number;
  nearest_road_distance?: number;
  conf_building_evidence?: number;
  conf_road_access?: number;
  conf_geometry_quality?: number;
  conf_topological_plausibility?: number;
  recommendation?: string;
}

export interface ParcelReviewSummary {
  total: number;
  unreviewed: number;
  accepted: number;
  edited: number;
  needs_review: number;
  rejected: number;
}

export interface ParcelReviewUpdateRequest {
  review_status: ReviewStatus;
  reviewer_note?: string | null;
  geometry?: GeoJSONGeometry | null;
  current_geometry?: GeoJSONGeometry | null;
}

export type ParcelFilterMode =
  | 'all'
  | 'unreviewed'
  | 'needs_review'
  | 'accepted'
  | 'edited'
  | 'rejected'
  | 'high_confidence'
  | 'moderate_confidence'
  | 'has_flags';

export interface RoadProperties {
  road_id: number;
  source: string;
  confidence: number;
  area_m2: number;
  length_m: number;
  width_estimate_m: number;
  geometry_valid: boolean;
}

export interface GeoJSONGeometry {
  type: 'Polygon' | 'MultiPolygon' | string;
  coordinates: any;
}

export interface GeoJSONFeature {
  type: 'Feature';
  id?: number | string;
  properties: BuildingProperties;
  geometry: GeoJSONGeometry;
}

export interface GeoJSONFeatureCollection {
  type: 'FeatureCollection';
  crs?: {
    type: string;
    properties: {
      name: string;
    };
  };
  features: GeoJSONFeature[];
}

export interface BuildingDetailResponse {
  id: number;
  area: number;
  perimeter: number;
  area_unit: string;
  perimeter_unit: string;
  centroid_x: number;
  centroid_y: number;
  confidence: number;
  min_confidence: number;
  is_georeferenced: boolean;
  geometry?: GeoJSONGeometry;
  feature?: GeoJSONFeature;
}

export type ProcessingStage =
  | 'idle'
  | 'uploading'
  | 'preparing'
  | 'segmenting'
  | 'extracting'
  | 'generating'
  | 'completed'
  | 'error';

export type VisualizationMode = 'normal' | 'confidence';
export type LayerViewMode = 'buildings' | 'overlay';
export type RoadSource = 'classical' | 'deep_learning' | 'hybrid';

export interface PipelineStatusResponse {
  analysis_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | string;
  progress: number;
  current_stage: string;
  completed_stages: string[];
  error?: string | null;
  technical_error?: string | null;
}

export interface AnalysisSummaryData {
  analysis_id: string;
  status: string;
  input_dimensions: [number, number];
  crs: string | null;
  georeferenced: boolean;
  processing_time: number;
  stage_timings: Record<string, number>;
  building_count: number;
  road_corridor_count: number;
  road_length_m: number;
  parcel_candidate_count: number;
  geometry_validity_pct: number;
  confidence_statistics: {
    mean: number;
    median: number;
    min: number;
    max: number;
  };
  flagged_parcel_count: number;
  review_status_counts: ParcelReviewSummary;
  disclaimer: string;
}

export interface AnalysisCreationResponse {
  analysis_id: string;
  status: string;
  current_stage: string;
  progress: number;
  message?: string;
}

