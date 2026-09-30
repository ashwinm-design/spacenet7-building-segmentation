/**
 * frontend/src/App.tsx
 *
 * Master Application Dashboard for SIH26012 AI Urban Mapping & Cadastral Feature Extraction.
 * Manages full lifecycle: health monitoring, file upload, inference triggering,
 * GeoJSON layer retrieval, interactive selection, responsive visualization,
 * and Phase 14B Human-in-the-Loop Parcel Verification workflow.
 */

import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import MapView from './components/MapView';
import {
  AnalysisResponse,
  GeoJSONFeatureCollection,
  BuildingDetailResponse,
  ProcessingStage,
  VisualizationMode,
  LayerViewMode,
  RoadSource,
  ParcelReviewItem,
  ParcelReviewSummary,
  ParcelFilterMode,
  ReviewStatus,
  PipelineStatusResponse,
  AnalysisSummaryData,
} from './types/analysis';
import {
  checkHealth,
  analyzeImage,
  runDemo,
  getAnalysis,
  getGeoJSON,
  getRoadsGeoJSON,
  getParcelsGeoJSON,
  getBuilding,
  getParcelReviews,
  getParcelReview,
  updateParcelReview,
  resetParcelGeometry,
  getPipelineStatus,
  getAnalysisFullSummary,
} from './api/analysis';

const DEFAULT_DEMO_ANALYSIS_ID = '8b8c724c7e';

export const App: React.FC = () => {
  // Backend & Service Status
  const [backendConnected, setBackendConnected] = useState<boolean>(false);
  const [stage, setStage] = useState<ProcessingStage>('idle');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatusResponse | null>(null);
  const [summaryData, setSummaryData] = useState<AnalysisSummaryData | null>(null);

  // Analysis Deliverables State
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [geojson, setGeojson] = useState<GeoJSONFeatureCollection | null>(null);
  const [roadsGeojson, setRoadsGeojson] = useState<GeoJSONFeatureCollection | null>(null);
  const [parcelsGeojson, setParcelsGeojson] = useState<GeoJSONFeatureCollection | null>(null);


  // Road extraction method toggle (Classical CV, Deep Learning, Hybrid Fusion)
  const [roadSource, setRoadSource] = useState<RoadSource>('hybrid');

  // Building Selection & Inspector State
  const [selectedBuildingId, setSelectedBuildingId] = useState<number | null>(null);
  const [selectedBuilding, setSelectedBuilding] = useState<BuildingDetailResponse | null>(null);

  // Phase 14B: Parcel Verification & Review State
  const [parcelReviews, setParcelReviews] = useState<Record<number, ParcelReviewItem>>({});
  const [reviewSummary, setReviewSummary] = useState<ParcelReviewSummary | null>(null);
  const [selectedParcelId, setSelectedParcelId] = useState<number | null>(null);
  const [selectedParcelReview, setSelectedParcelReview] = useState<ParcelReviewItem | null>(null);
  const [parcelFilterMode, setParcelFilterMode] = useState<ParcelFilterMode>('all');
  const [isEditingGeometry, setIsEditingGeometry] = useState<boolean>(false);
  const [showOriginalGeometry, setShowOriginalGeometry] = useState<boolean>(false);

  // Visualization Options
  const [visualizationMode, setVisualizationMode] = useState<VisualizationMode>('normal');
  const [layerMode, setLayerMode] = useState<LayerViewMode>('buildings');

  // 1. Health monitoring loop
  const verifyBackend = useCallback(async () => {
    try {
      const res = await checkHealth();
      if (res && res.status === 'ok') {
        setBackendConnected(true);
      } else {
        setBackendConnected(false);
      }
    } catch {
      setBackendConnected(false);
    }
  }, []);

  useEffect(() => {
    verifyBackend();
    const interval = setInterval(verifyBackend, 10000);
    return () => clearInterval(interval);
  }, [verifyBackend]);

  // Helper to fetch parcel reviews for active analysis
  const fetchParcelReviewsData = useCallback(async (analysisId: string) => {
    try {
      const data = await getParcelReviews(analysisId);
      if (data && data.reviews) {
        const reviewMap: Record<number, ParcelReviewItem> = {};
        data.reviews.forEach((r) => {
          reviewMap[r.candidate_id] = r;
        });
        setParcelReviews(reviewMap);
        setReviewSummary(data.summary);
      }
    } catch (err) {
      console.warn('Failed to fetch parcel reviews:', err);
    }
  }, []);

  // 2. Auto-load existing demo analysis (8b8c724c7e) on mount if available
  useEffect(() => {
    let isCancelled = false;

    const loadInitialDemo = async () => {
      try {
        const [demoAnalysis, summary] = await Promise.all([
          getAnalysis(DEFAULT_DEMO_ANALYSIS_ID),
          getAnalysisFullSummary(DEFAULT_DEMO_ANALYSIS_ID).catch(() => null),
        ]);
        if (demoAnalysis && !isCancelled) {
          setAnalysis(demoAnalysis);
          if (summary) setSummaryData(summary);
          setStage('completed');

          const [bldgs, roads, parcels] = await Promise.all([
            getGeoJSON(DEFAULT_DEMO_ANALYSIS_ID).catch(() => null),
            getRoadsGeoJSON(DEFAULT_DEMO_ANALYSIS_ID, 'hybrid').catch(() => null),
            getParcelsGeoJSON(DEFAULT_DEMO_ANALYSIS_ID, 'hybrid').catch(() => null),
          ]);

          if (!isCancelled) {
            setGeojson(bldgs);
            setRoadsGeojson(roads);
            setParcelsGeojson(parcels);
            fetchParcelReviewsData(DEFAULT_DEMO_ANALYSIS_ID);
          }
        }
      } catch (err) {
        // Fallback: wait for user interaction
      }
    };

    loadInitialDemo();
    return () => {
      isCancelled = true;
    };
  }, [fetchParcelReviewsData]);

  // Helper: Poll real background pipeline execution progress
  const startPipelinePolling = useCallback(
    (analysisId: string) => {
      let isCancelled = false;
      let attempts = 0;
      const maxAttempts = 120; // 3 minutes max

      const poll = async () => {
        if (isCancelled) return;
        attempts += 1;

        try {
          const statusData = await getPipelineStatus(analysisId);
          if (isCancelled) return;

          setPipelineStatus(statusData);

          const st = statusData.current_stage;
          if (st === 'VALIDATING' || st === 'PREPROCESSING') {
            setStage('preparing');
          } else if (st === 'BUILDING_INFERENCE') {
            setStage('segmenting');
          } else if (st === 'BUILDING_POLYGONIZATION') {
            setStage('extracting');
          } else if (
            [
              'ROAD_EXTRACTION',
              'ROAD_FUSION',
              'PARCEL_GENERATION',
              'PARCEL_INTELLIGENCE',
              'READY_FOR_REVIEW',
            ].includes(st)
          ) {
            setStage('generating');
          }

          if (statusData.status === 'completed') {
            setStage('completed');

            // Fetch deliverables in parallel
            const [meta, summary, bldgs, roads, parcels] = await Promise.all([
              getAnalysis(analysisId).catch(() => null),
              getAnalysisFullSummary(analysisId).catch(() => null),
              getGeoJSON(analysisId).catch(() => null),
              getRoadsGeoJSON(analysisId, roadSource).catch(() => null),
              getParcelsGeoJSON(analysisId, roadSource).catch(() => null),
            ]);

            if (!isCancelled) {
              if (meta) setAnalysis(meta);
              if (summary) setSummaryData(summary);
              setGeojson(bldgs);
              setRoadsGeojson(roads);
              setParcelsGeojson(parcels);
              fetchParcelReviewsData(analysisId);
            }
            return;
          }

          if (statusData.status === 'failed') {
            setStage('error');
            setErrorMsg(statusData.error || 'Pipeline execution failed.');
            return;
          }

          if (attempts >= maxAttempts) {
            setStage('error');
            setErrorMsg('Analysis timed out after 3 minutes.');
            return;
          }

          // Continue polling
          setTimeout(poll, 1500);
        } catch (err: any) {
          if (!isCancelled) {
            if (attempts >= maxAttempts) {
              setStage('error');
              setErrorMsg('Failed to poll analysis progress.');
            } else {
              setTimeout(poll, 2000);
            }
          }
        }
      };

      poll();
      return () => {
        isCancelled = true;
      };
    },
    [fetchParcelReviewsData, roadSource]
  );

  // 3. Handle File Analysis
  const handleAnalyze = async (file: File, roi?: string) => {
    setErrorMsg(null);
    setSelectedBuildingId(null);
    setSelectedBuilding(null);
    setSelectedParcelId(null);
    setSelectedParcelReview(null);
    setPipelineStatus(null);
    setStage('uploading');

    try {
      const result = await analyzeImage(file, roi, true);
      const analysisId = result.analysis_id;

      setAnalysis({
        analysis_id: analysisId,
        status: 'queued',
        building_count: 0,
        total_area: 0,
        area_unit: 'm²',
        average_building_area: 0,
        average_confidence: 0,
        processing_time: 0,
        crs: null,
        georeferenced: false,
        geojson_url: '',
        statistics_url: '',
        overlay_url: '',
      });

      startPipelinePolling(analysisId);
    } catch (err: any) {
      const detail =
        err.response?.data?.detail || err.message || 'Failed to submit analysis.';
      setErrorMsg(detail);
      setStage('error');
    }
  };

  // 4. Handle One-Click Demo Mode
  const handleTryDemo = async () => {
    setErrorMsg(null);
    setSelectedBuildingId(null);
    setSelectedBuilding(null);
    setSelectedParcelId(null);
    setSelectedParcelReview(null);
    setPipelineStatus(null);
    setStage('uploading');

    try {
      const result = await runDemo(true);
      const analysisId = result.analysis_id;

      setAnalysis({
        analysis_id: analysisId,
        status: 'queued',
        building_count: 0,
        total_area: 0,
        area_unit: 'm²',
        average_building_area: 0,
        average_confidence: 0,
        processing_time: 0,
        crs: 'EPSG:32732',
        georeferenced: true,
        geojson_url: '',
        statistics_url: '',
        overlay_url: '',
      });

      startPipelinePolling(analysisId);
    } catch (err: any) {
      const detail =
        err.response?.data?.detail || err.message || 'Failed to execute demo.';
      setErrorMsg(detail);
      setStage('error');
    }
  };


  // 5. Re-fetch roads and parcels when road source changes (Classical, DL, Hybrid)
  useEffect(() => {
    if (!analysis) return;
    let isCancelled = false;

    const fetchMethodLayers = async () => {
      try {
        const [roadsData, parcelsData] = await Promise.all([
          getRoadsGeoJSON(analysis.analysis_id, roadSource).catch(() => null),
          getParcelsGeoJSON(analysis.analysis_id, roadSource).catch(() => null),
        ]);
        if (!isCancelled) {
          setRoadsGeojson(roadsData);
          setParcelsGeojson(parcelsData);
        }
      } catch (err) {
        console.warn('Failed to switch road source:', err);
      }
    };

    fetchMethodLayers();
    return () => {
      isCancelled = true;
    };
  }, [analysis, roadSource]);

  // 6. Handle Building Selection
  const handleSelectBuilding = async (buildingId: number) => {
    if (!analysis) return;
    setSelectedBuildingId(buildingId);
    setSelectedParcelId(null);
    setSelectedParcelReview(null);
    setIsEditingGeometry(false);

    try {
      const detail = await getBuilding(analysis.analysis_id, buildingId);
      setSelectedBuilding(detail);
    } catch (err) {
      console.warn('Could not fetch single building detail:', err);
    }
  };

  const handleCloseBuilding = () => {
    setSelectedBuildingId(null);
    setSelectedBuilding(null);
  };

  // 7. Handle Parcel Selection (Phase 14B)
  const handleSelectParcel = async (parcelId: number) => {
    if (!analysis) return;
    setSelectedParcelId(parcelId);
    setSelectedBuildingId(null);
    setSelectedBuilding(null);
    setIsEditingGeometry(false);
    setShowOriginalGeometry(false);

    // Check local cache first for instant display
    if (parcelReviews[parcelId]) {
      setSelectedParcelReview(parcelReviews[parcelId]);
    }

    try {
      const reviewItem = await getParcelReview(analysis.analysis_id, parcelId);
      setSelectedParcelReview(reviewItem);
      setParcelReviews((prev) => ({ ...prev, [parcelId]: reviewItem }));
    } catch (err) {
      console.warn('Could not fetch single parcel review:', err);
    }
  };

  const handleCloseParcelReview = () => {
    setSelectedParcelId(null);
    setSelectedParcelReview(null);
    setIsEditingGeometry(false);
    setShowOriginalGeometry(false);
  };

  // 8. Handle Parcel Review Status Decision (Accept, Needs Review, Reject, Save Note)
  const handleUpdateParcelStatus = async (status: ReviewStatus, note?: string) => {
    if (!analysis || !selectedParcelId) return;

    try {
      const updated = await updateParcelReview(analysis.analysis_id, selectedParcelId, {
        review_status: status,
        reviewer_note: note,
      });

      setSelectedParcelReview(updated);
      setParcelReviews((prev) => ({ ...prev, [selectedParcelId]: updated }));

      // Refresh summary & parcels layer so status color updates immediately
      const [sum, refreshedParcels] = await Promise.all([
        getParcelReviews(analysis.analysis_id).then((r) => r.summary),
        getParcelsGeoJSON(analysis.analysis_id, roadSource).catch(() => null),
      ]);
      setReviewSummary(sum);
      if (refreshedParcels) setParcelsGeojson(refreshedParcels);
    } catch (err: any) {
      console.error('Failed to update parcel review:', err);
      alert(err.response?.data?.detail || 'Failed to update review status.');
    }
  };

  // 9. Handle Saving Edited Geometry (Step 5)
  const handleSaveEditedGeometry = async (candidateId: number, newGeometry: any) => {
    if (!analysis) return;

    try {
      const updated = await updateParcelReview(analysis.analysis_id, candidateId, {
        review_status: 'edited',
        reviewer_note: selectedParcelReview?.reviewer_note,
        geometry: newGeometry,
      });

      setSelectedParcelReview(updated);
      setParcelReviews((prev) => ({ ...prev, [candidateId]: updated }));
      setIsEditingGeometry(false);

      // Refresh summary & parcels layer
      const [sum, refreshedParcels] = await Promise.all([
        getParcelReviews(analysis.analysis_id).then((r) => r.summary),
        getParcelsGeoJSON(analysis.analysis_id, roadSource).catch(() => null),
      ]);
      setReviewSummary(sum);
      if (refreshedParcels) setParcelsGeojson(refreshedParcels);
    } catch (err: any) {
      console.error('Failed to save edited geometry:', err);
      alert(err.response?.data?.detail || 'Failed to save edited geometry.');
    }
  };

  // 10. Handle Resetting Edited Geometry (Step 5 & Step 10)
  const handleResetGeometry = async () => {
    if (!analysis || !selectedParcelId) return;

    try {
      const updated = await resetParcelGeometry(analysis.analysis_id, selectedParcelId);
      setSelectedParcelReview(updated);
      setParcelReviews((prev) => ({ ...prev, [selectedParcelId]: updated }));
      setIsEditingGeometry(false);
      setShowOriginalGeometry(false);

      // Refresh summary & layer
      const [sum, refreshedParcels] = await Promise.all([
        getParcelReviews(analysis.analysis_id).then((r) => r.summary),
        getParcelsGeoJSON(analysis.analysis_id, roadSource).catch(() => null),
      ]);
      setReviewSummary(sum);
      if (refreshedParcels) setParcelsGeojson(refreshedParcels);
    } catch (err: any) {
      console.error('Failed to reset parcel geometry:', err);
      alert(err.response?.data?.detail || 'Failed to reset parcel geometry.');
    }
  };

  // 11. Review Next Algorithm (Step 9)
  // Priority: 1. needs_review candidates first
  //           2. lowest confidence first
  //           3. deterministic candidate ID ordering as tie-breaker
  const handleReviewNext = () => {
    const reviewsList = Object.values(parcelReviews);
    if (reviewsList.length === 0) return;

    // Filter unreviewed and needs_review candidates
    const needsReviewList = reviewsList.filter((r) => r.review_status === 'needs_review');
    const unreviewedList = reviewsList.filter((r) => r.review_status === 'unreviewed');

    let pool = needsReviewList.length > 0
      ? needsReviewList
      : unreviewedList.length > 0
      ? unreviewedList
      : reviewsList;

    // Sort by lowest confidence first, tie-breaker: candidate_id ascending
    pool.sort((a, b) => {
      if (a.confidence !== b.confidence) {
        return a.confidence - b.confidence;
      }
      return a.candidate_id - b.candidate_id;
    });

    const nextCandidate = pool[0];
    if (nextCandidate) {
      handleSelectParcel(nextCandidate.candidate_id);
    }
  };

  const isProcessing = ['uploading', 'preparing', 'segmenting', 'extracting', 'generating'].includes(
    stage
  );

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-gis-bg text-gis-textPrimary font-sans">
      {/* Top Navigation & Status Header */}
      <Header
        backendConnected={backendConnected}
        onTryDemo={handleTryDemo}
        isProcessing={isProcessing}
      />

      {/* Main Workspace: Sidebar + Map */}
      <div className="flex flex-1 overflow-hidden relative">
        <Sidebar
          onAnalyze={handleAnalyze}
          isProcessing={isProcessing}
          backendConnected={backendConnected}
          stage={stage}
          error={errorMsg}
          analysis={analysis}
          selectedBuilding={selectedBuilding}
          onCloseBuilding={handleCloseBuilding}
          pipelineStatus={pipelineStatus}
          summaryData={summaryData}
          onOpenWebGIS={() => {
            setLayerMode('buildings');
            setSelectedBuildingId(null);
            setSelectedBuilding(null);
          }}
          reviewSummary={reviewSummary}
          selectedParcelReview={selectedParcelReview}
          onCloseParcelReview={handleCloseParcelReview}
          onUpdateParcelStatus={handleUpdateParcelStatus}
          isEditingGeometry={isEditingGeometry}
          onStartEditGeometry={() => setIsEditingGeometry(true)}
          onCancelEditGeometry={() => setIsEditingGeometry(false)}
          onResetGeometry={handleResetGeometry}
          showOriginalGeometry={showOriginalGeometry}
          onToggleShowOriginal={setShowOriginalGeometry}
          onReviewNext={handleReviewNext}
        />

        <main className="flex-1 relative flex flex-col h-full overflow-hidden">
          <MapView
            geojson={geojson}
            roadsGeojson={roadsGeojson}
            parcelsGeojson={parcelsGeojson}
            analysis={analysis}
            selectedBuildingId={selectedBuildingId}
            onSelectBuilding={handleSelectBuilding}
            selectedParcelId={selectedParcelId}
            onSelectParcel={handleSelectParcel}
            visualizationMode={visualizationMode}
            onChangeVisualizationMode={setVisualizationMode}
            layerMode={layerMode}
            onChangeLayerMode={setLayerMode}
            roadSource={roadSource}
            onChangeRoadSource={setRoadSource}
            parcelFilterMode={parcelFilterMode}
            onChangeParcelFilterMode={setParcelFilterMode}
            isEditingGeometry={isEditingGeometry}
            onSaveEditedGeometry={handleSaveEditedGeometry}
            onCancelEditGeometry={() => setIsEditingGeometry(false)}
            showOriginalGeometry={showOriginalGeometry}
            originalGeometry={selectedParcelReview?.original_geometry || null}
          />
        </main>
      </div>
    </div>
  );
};

export default App;
