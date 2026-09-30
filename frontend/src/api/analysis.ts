/**
 * frontend/src/api/analysis.ts
 *
 * Typed API service functions for interacting with SIH26012 FastAPI endpoints.
 */

import { apiClient, API_BASE_URL } from './client';
import {
  HealthResponse,
  AnalysisResponse,
  GeoJSONFeatureCollection,
  BuildingDetailResponse,
  ParcelReviewItem,
  ParcelReviewSummary,
  ParcelReviewUpdateRequest,
  PipelineStatusResponse,
  AnalysisSummaryData,
  AnalysisCreationResponse,
} from '../types/analysis';

/**
 * Health check endpoint.
 */
export async function checkHealth(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>('/api/health');
  return response.data;
}

/**
 * Analyze an uploaded aerial image with optional background queue execution.
 */
export async function analyzeImage(
  file: File,
  roi?: string,
  background: boolean = true,
  onUploadProgress?: (progressEvent: any) => void
): Promise<AnalysisCreationResponse | AnalysisResponse> {
  const formData = new FormData();
  formData.append('file', file);
  if (roi && roi.trim()) {
    formData.append('roi', roi.trim());
  }
  if (background) {
    formData.append('background', 'true');
  }

  const response = await apiClient.post<AnalysisCreationResponse | AnalysisResponse>('/api/analyze', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress,
  });

  return response.data;
}

/**
 * Run inference on the locked Open Cities GeoTIFF demo ROI (3072, 2048, 4096, 3072).
 */
export async function runDemo(background: boolean = true): Promise<AnalysisCreationResponse | AnalysisResponse> {
  const response = await apiClient.post<AnalysisCreationResponse | AnalysisResponse>(
    `/api/demo?background=${background}`
  );
  return response.data;
}


/**
 * Retrieve metadata for a completed analysis.
 */
export async function getAnalysis(analysisId: string): Promise<AnalysisResponse> {
  const response = await apiClient.get<AnalysisResponse>(`/api/results/${analysisId}`);
  return response.data;
}

/**
 * Fetch the raw GeoJSON FeatureCollection.
 */
export async function getGeoJSON(analysisId: string): Promise<GeoJSONFeatureCollection> {
  const response = await apiClient.get<GeoJSONFeatureCollection>(`/api/results/${analysisId}/geojson`);
  return response.data;
}

/**
 * Fetch extracted physical road corridors GeoJSON (source: 'classical' | 'deep_learning' | 'hybrid').
 */
export async function getRoadsGeoJSON(
  analysisId: string,
  source: 'classical' | 'deep_learning' | 'hybrid' = 'classical'
): Promise<GeoJSONFeatureCollection> {
  const response = await apiClient.get<GeoJSONFeatureCollection>(
    `/api/results/${analysisId}/roads?source=${source}`
  );
  return response.data;
}

/**
 * Fetch road-constrained preliminary parcel candidates GeoJSON (source: 'classical' | 'deep_learning' | 'hybrid').
 */
export async function getParcelsGeoJSON(
  analysisId: string,
  source: 'classical' | 'deep_learning' | 'hybrid' = 'classical'
): Promise<GeoJSONFeatureCollection> {
  const response = await apiClient.get<GeoJSONFeatureCollection>(
    `/api/results/${analysisId}/parcels?source=${source}`
  );
  return response.data;
}

/**
 * Fetch detailed attributes for a specific building footprint.
 */
export async function getBuilding(
  analysisId: string,
  buildingId: number
): Promise<BuildingDetailResponse> {
  const response = await apiClient.get<BuildingDetailResponse>(
    `/api/results/${analysisId}/buildings/${buildingId}`
  );
  return response.data;
}

/**
 * Helper URL getters for direct browser downloads.
 */
export function getGeoJsonDownloadUrl(analysisId: string): string {
  return `${API_BASE_URL}/api/results/${analysisId}/geojson`;
}

export function getStatisticsDownloadUrl(analysisId: string): string {
  return `${API_BASE_URL}/api/results/${analysisId}/statistics`;
}

export function getOverlayDownloadUrl(analysisId: string): string {
  return `${API_BASE_URL}/api/results/${analysisId}/overlay`;
}

/**
 * Fetch all parcel reviews and breakdown summary for an analysis job.
 */
export async function getParcelReviews(
  analysisId: string
): Promise<{ analysis_id: string; summary: ParcelReviewSummary; reviews: ParcelReviewItem[] }> {
  const response = await apiClient.get<{
    analysis_id: string;
    summary: ParcelReviewSummary;
    reviews: ParcelReviewItem[];
  }>(`/api/results/${analysisId}/parcel-reviews`);
  return response.data;
}

/**
 * Fetch review summary counts (total, unreviewed, accepted, edited, needs_review, rejected).
 */
export async function getParcelReviewSummary(
  analysisId: string
): Promise<ParcelReviewSummary> {
  const response = await apiClient.get<ParcelReviewSummary>(
    `/api/results/${analysisId}/parcel-reviews/summary`
  );
  return response.data;
}

/**
 * Fetch review state, uncertainty metrics, and geometries for a single parcel candidate.
 */
export async function getParcelReview(
  analysisId: string,
  candidateId: number
): Promise<ParcelReviewItem> {
  const response = await apiClient.get<ParcelReviewItem>(
    `/api/results/${analysisId}/parcel-reviews/${candidateId}`
  );
  return response.data;
}

/**
 * Update review state (accept, flag, edit, or reject) with optional edited geometry and reviewer notes.
 */
export async function updateParcelReview(
  analysisId: string,
  candidateId: number,
  data: ParcelReviewUpdateRequest
): Promise<ParcelReviewItem> {
  const response = await apiClient.post<ParcelReviewItem>(
    `/api/results/${analysisId}/parcel-reviews/${candidateId}`,
    data
  );
  return response.data;
}

/**
 * Reset edited parcel geometry back to the original AI hypothesis.
 */
export async function resetParcelGeometry(
  analysisId: string,
  candidateId: number
): Promise<ParcelReviewItem> {
  const response = await apiClient.post<ParcelReviewItem>(
    `/api/results/${analysisId}/parcel-reviews/${candidateId}/reset`
  );
  return response.data;
}

/**
 * Helper URL getter for exporting GIS-ready reviewed parcels GeoJSON.
 */
export function getParcelReviewExportUrl(analysisId: string): string {
  return `${API_BASE_URL}/api/results/${analysisId}/parcel-reviews/export`;
}

/**
 * Poll real-time pipeline execution progress and active stage.
 */
export async function getPipelineStatus(analysisId: string): Promise<PipelineStatusResponse> {
  const response = await apiClient.get<PipelineStatusResponse>(
    `/api/analysis/${analysisId}/status`
  );
  return response.data;
}

/**
 * Fetch comprehensive analysis summary JSON.
 */
export async function getAnalysisFullSummary(analysisId: string): Promise<AnalysisSummaryData> {
  const response = await apiClient.get<AnalysisSummaryData>(
    `/api/results/${analysisId}/summary`
  );
  return response.data;
}

export function getSummaryDownloadUrl(analysisId: string): string {
  return `${API_BASE_URL}/api/results/${analysisId}/summary`;
}

export function getAuditDownloadUrl(analysisId: string): string {
  return `${API_BASE_URL}/api/results/${analysisId}/parcel-reviews/audit`;
}

export function getRoadsDownloadUrl(analysisId: string, source: string = 'hybrid'): string {
  return `${API_BASE_URL}/api/results/${analysisId}/roads?source=${source}`;
}

export function getParcelsDownloadUrl(analysisId: string, source: string = 'hybrid'): string {
  return `${API_BASE_URL}/api/results/${analysisId}/parcels?source=${source}`;
}


