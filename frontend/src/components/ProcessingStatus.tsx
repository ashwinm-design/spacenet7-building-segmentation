/**
 * frontend/src/components/ProcessingStatus.tsx
 *
 * Real-time processing progress bar, sequential pipeline stage checklist,
 * and completion summary card for SIH26012 Phase 15.
 */

import React, { useState } from 'react';
import {
  Loader2,
  CheckCircle2,
  AlertTriangle,
  Layers,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Compass,
  FileCheck,
  Check,
} from 'lucide-react';
import { PipelineStatusResponse, AnalysisSummaryData, ProcessingStage } from '../types/analysis';

interface ProcessingStatusProps {
  stage: ProcessingStage;
  error?: string | null;
  pipelineStatus?: PipelineStatusResponse | null;
  analysisId?: string | null;
  summaryData?: AnalysisSummaryData | null;
  onOpenWebGIS?: () => void;
}

interface StageStep {
  key: string;
  label: string;
  order: number;
}

const PIPELINE_STEPS: StageStep[] = [
  { key: 'VALIDATING', label: 'Input validated', order: 1 },
  { key: 'BUILDING_INFERENCE', label: 'Building extraction', order: 2 },
  { key: 'BUILDING_POLYGONIZATION', label: 'Building polygonization', order: 3 },
  { key: 'ROAD_EXTRACTION', label: 'Road corridor extraction', order: 4 },
  { key: 'ROAD_FUSION', label: 'Hybrid road extraction', order: 5 },
  { key: 'PARCEL_GENERATION', label: 'Parcel generation', order: 6 },
  { key: 'PARCEL_INTELLIGENCE', label: 'Parcel intelligence', order: 7 },
  { key: 'READY_FOR_REVIEW', label: 'Verification queue ready', order: 8 },
];

export const ProcessingStatus: React.FC<ProcessingStatusProps> = ({
  stage,
  error,
  pipelineStatus,
  analysisId,
  summaryData,
  onOpenWebGIS,
}) => {
  const [showLogs, setShowLogs] = useState(false);

  if (stage === 'idle' && !pipelineStatus) return null;

  const isCompleted = stage === 'completed' || pipelineStatus?.status === 'completed';
  const isFailed = stage === 'error' || pipelineStatus?.status === 'failed';
  const isWorking = !isCompleted && !isFailed;

  const currentStage = pipelineStatus?.current_stage || stage.toUpperCase();
  const completedStages = pipelineStatus?.completed_stages || [];
  const progressPct = pipelineStatus?.progress ?? (isCompleted ? 100 : 0);

  // Helper to determine step state
  const getStepState = (stepKey: string, stepOrder: number) => {
    if (completedStages.includes(stepKey)) return 'done';
    if (currentStage === stepKey) return 'active';

    // Find current stage order
    const currentStepObj = PIPELINE_STEPS.find((s) => s.key === currentStage);
    if (currentStepObj && currentStepObj.order > stepOrder) return 'done';
    if (isCompleted) return 'done';
    return 'pending';
  };

  return (
    <div className="bg-gis-card rounded-lg border border-gis-border p-4 shadow-md space-y-3">
      {/* Header with Analysis ID */}
      <div className="flex items-center justify-between border-b border-gis-border pb-2">
        <div className="flex items-center space-x-2">
          {isWorking ? (
            <Loader2 className="w-4 h-4 text-gis-cyan animate-spin" />
          ) : isCompleted ? (
            <CheckCircle2 className="w-4 h-4 text-gis-success" />
          ) : (
            <AlertTriangle className="w-4 h-4 text-gis-danger" />
          )}
          <span className="font-bold text-white text-xs tracking-wider uppercase">
            {isCompleted
              ? 'ANALYSIS COMPLETE'
              : isFailed
              ? 'ANALYSIS FAILED'
              : analysisId
              ? `ANALYSIS ${analysisId}`
              : 'ANALYZING IMAGERY'}
          </span>
        </div>
        {analysisId && (
          <span className="text-[10px] font-mono text-gis-textMuted bg-gis-surface px-1.5 py-0.5 rounded border border-gis-border">
            {analysisId}
          </span>
        )}
      </div>

      {/* Real Stage Progress Bar */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-[11px]">
          <span className="text-gis-textMuted">
            {isWorking ? 'Processing pipeline...' : isCompleted ? 'All stages completed' : 'Stopped'}
          </span>
          <span className="font-mono font-bold text-gis-cyan">{progressPct}%</span>
        </div>
        <div className="w-full bg-gis-surface rounded-full h-2 overflow-hidden border border-gis-border/50">
          <div
            className={`h-full transition-all duration-500 rounded-full ${
              isFailed
                ? 'bg-gis-danger'
                : isCompleted
                ? 'bg-gis-success'
                : 'bg-gradient-to-r from-gis-accent via-gis-cyan to-sky-400'
            }`}
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </div>

      {/* Sequential Pipeline Stages Checklist */}
      <div className="bg-gis-surface/50 rounded-md p-2.5 border border-gis-border/70 space-y-1.5 text-xs">
        {PIPELINE_STEPS.map((step) => {
          const state = getStepState(step.key, step.order);
          return (
            <div key={step.key} className="flex items-center space-x-2 text-[11px]">
              {state === 'done' ? (
                <span className="w-3.5 h-3.5 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-[10px]">
                  ✓
                </span>
              ) : state === 'active' ? (
                <span className="w-3.5 h-3.5 rounded-full bg-gis-cyan/20 text-gis-cyan flex items-center justify-center animate-pulse text-[11px]">
                  ●
                </span>
              ) : (
                <span className="w-3.5 h-3.5 rounded-full text-slate-500 flex items-center justify-center text-[10px]">
                  ○
                </span>
              )}
              <span
                className={
                  state === 'done'
                    ? 'text-gis-textPrimary'
                    : state === 'active'
                    ? 'text-white font-semibold'
                    : 'text-gis-textMuted'
                }
              >
                {step.label}
              </span>
            </div>
          );
        })}
      </div>

      {/* Error Display if Failed */}
      {isFailed && (
        <div className="p-2.5 rounded bg-gis-danger/10 border border-gis-danger/30 text-gis-danger text-xs space-y-1">
          <div className="font-semibold flex items-center space-x-1.5">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>Failed Stage: {pipelineStatus?.current_stage || 'Unknown'}</span>
          </div>
          <p className="text-[11px] opacity-90">{pipelineStatus?.error || error || 'Pipeline execution failed.'}</p>
        </div>
      )}

      {/* Completed Results Summary Box */}
      {isCompleted && summaryData && (
        <div className="bg-gis-surface/80 rounded-md p-3 border border-emerald-500/30 space-y-2 text-xs">
          <div className="font-semibold text-emerald-400 text-[11px] uppercase tracking-wider flex items-center justify-between">
            <span>Extraction Deliverables</span>
            <span className="font-mono text-gis-textMuted">{summaryData.processing_time}s</span>
          </div>
          <div className="grid grid-cols-2 gap-1.5 text-[11px]">
            <div className="text-gis-textMuted">
              Buildings: <span className="text-white font-bold">{summaryData.building_count}</span>
            </div>
            <div className="text-gis-textMuted">
              Road network: <span className="text-white font-bold">{summaryData.road_length_m} m</span>
            </div>
            <div className="text-gis-textMuted">
              Parcels: <span className="text-white font-bold">{summaryData.parcel_candidate_count}</span>
            </div>
            <div className="text-gis-textMuted">
              Geometry validity: <span className="text-emerald-400 font-bold">{summaryData.geometry_validity_pct}%</span>
            </div>
          </div>

          {onOpenWebGIS && (
            <button
              type="button"
              onClick={onOpenWebGIS}
              className="w-full mt-2 py-1.5 px-3 bg-gis-accent hover:bg-gis-accent/80 text-white rounded font-bold text-xs transition-colors flex items-center justify-center space-x-1.5 shadow"
            >
              <Compass className="w-3.5 h-3.5" />
              <span>OPEN WEBGIS</span>
            </button>
          )}
        </div>
      )}

      {/* Collapsible Technical Status Details */}
      <div>
        <button
          type="button"
          onClick={() => setShowLogs(!showLogs)}
          className="text-[10px] text-gis-textMuted hover:text-white flex items-center space-x-1 transition-colors"
        >
          {showLogs ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          <span>{showLogs ? 'Hide Pipeline Logs' : 'View Pipeline Logs'}</span>
        </button>

        {showLogs && (
          <div className="mt-2 p-2 bg-gis-bg rounded border border-gis-border/60 text-[10px] font-mono text-gis-textMuted space-y-1">
            <div>Status: {pipelineStatus?.status || stage}</div>
            <div>Stage: {currentStage}</div>
            <div>Completed: [{completedStages.join(', ')}]</div>
            {pipelineStatus?.technical_error && (
              <div className="text-gis-danger">Exception: {pipelineStatus.technical_error}</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ProcessingStatus;
