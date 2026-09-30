/**
 * frontend/src/components/Sidebar.tsx
 *
 * Left-side control panel housing Upload, Processing, Dashboard Summary,
 * Building Details, Parcel Verification & Inspection (Phase 14B),
 * Review Queue, and Deliverables Export Center (Phase 15).
 */

import React from 'react';
import UploadPanel from './UploadPanel';
import ProcessingStatus from './ProcessingStatus';
import AnalysisStats from './AnalysisStats';
import BuildingPanel from './BuildingPanel';
import ExportPanel from './ExportPanel';
import ParcelInspectionPanel from './ParcelInspectionPanel';
import ReviewQueuePanel from './ReviewQueuePanel';
import {
  AnalysisResponse,
  BuildingDetailResponse,
  ProcessingStage,
  ParcelReviewItem,
  ParcelReviewSummary,
  ReviewStatus,
  PipelineStatusResponse,
  AnalysisSummaryData,
} from '../types/analysis';

interface SidebarProps {
  onAnalyze: (file: File, roi?: string) => void;
  isProcessing: boolean;
  backendConnected: boolean;
  stage: ProcessingStage;
  error: string | null;
  analysis: AnalysisResponse | null;
  selectedBuilding: BuildingDetailResponse | null;
  onCloseBuilding: () => void;

  // Phase 15 Pipeline & Summary Props
  pipelineStatus?: PipelineStatusResponse | null;
  summaryData?: AnalysisSummaryData | null;
  onOpenWebGIS?: () => void;

  // Phase 14B Parcel Review Props
  reviewSummary: ParcelReviewSummary | null;
  selectedParcelReview: ParcelReviewItem | null;
  onCloseParcelReview: () => void;
  onUpdateParcelStatus: (status: ReviewStatus, note?: string) => Promise<void>;
  isEditingGeometry: boolean;
  onStartEditGeometry: () => void;
  onCancelEditGeometry: () => void;
  onResetGeometry: () => Promise<void>;
  showOriginalGeometry: boolean;
  onToggleShowOriginal: (show: boolean) => void;
  onReviewNext: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  onAnalyze,
  isProcessing,
  backendConnected,
  stage,
  error,
  analysis,
  selectedBuilding,
  onCloseBuilding,
  pipelineStatus,
  summaryData,
  onOpenWebGIS,
  reviewSummary,
  selectedParcelReview,
  onCloseParcelReview,
  onUpdateParcelStatus,
  isEditingGeometry,
  onStartEditGeometry,
  onCancelEditGeometry,
  onResetGeometry,
  showOriginalGeometry,
  onToggleShowOriginal,
  onReviewNext,
}) => {
  return (
    <aside className="w-96 flex-shrink-0 bg-gis-bg border-r border-gis-border flex flex-col h-full overflow-hidden z-20">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Upload Control */}
        <UploadPanel
          onAnalyze={onAnalyze}
          isProcessing={isProcessing}
          disabled={!backendConnected}
        />

        {/* Real Processing Pipeline Progress & Stages Checklist */}
        <ProcessingStatus
          stage={stage}
          error={error}
          pipelineStatus={pipelineStatus}
          analysisId={analysis?.analysis_id}
          summaryData={summaryData}
          onOpenWebGIS={onOpenWebGIS}
        />

        {/* Phase 15 Dashboard Summary Statistics */}
        <AnalysisStats
          data={analysis}
          summaryData={summaryData}
          reviewSummary={reviewSummary}
        />

        {/* Selected Building Details */}
        <BuildingPanel building={selectedBuilding} onClose={onCloseBuilding} />

        {/* Phase 14B: Dedicated Parcel Intelligence & Verification Panel */}
        {selectedParcelReview && (
          <ParcelInspectionPanel
            review={selectedParcelReview}
            onClose={onCloseParcelReview}
            onUpdateStatus={onUpdateParcelStatus}
            isEditingGeometry={isEditingGeometry}
            onStartEditGeometry={onStartEditGeometry}
            onCancelEditGeometry={onCancelEditGeometry}
            onResetGeometry={onResetGeometry}
            showOriginalGeometry={showOriginalGeometry}
            onToggleShowOriginal={onToggleShowOriginal}
          />
        )}

        {/* Phase 14B: Review Queue Summary */}
        {reviewSummary && analysis && (
          <ReviewQueuePanel
            summary={reviewSummary}
            analysis={analysis}
            onReviewNext={onReviewNext}
            disabled={isProcessing}
          />
        )}

        {/* Phase 15: Deliverables Export Center */}
        <ExportPanel analysis={analysis} />
      </div>
    </aside>
  );
};

export default Sidebar;
