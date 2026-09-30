/**
 * frontend/src/components/ReviewQueuePanel.tsx
 *
 * Parcel Review Queue Summary and Next Reviewer Action panel (Phase 14B).
 * Displays real-time breakdown of parcel candidates and prioritizes review next.
 */

import React from 'react';
import {
  ClipboardCheck,
  Clock,
  AlertTriangle,
  CheckCircle,
  Edit3,
  XCircle,
  ArrowRight,
  Download,
} from 'lucide-react';
import { ParcelReviewSummary, AnalysisResponse } from '../types/analysis';
import { getParcelReviewExportUrl } from '../api/analysis';

interface ReviewQueuePanelProps {
  summary: ParcelReviewSummary | null;
  analysis: AnalysisResponse | null;
  onReviewNext: () => void;
  disabled?: boolean;
}

export const ReviewQueuePanel: React.FC<ReviewQueuePanelProps> = ({
  summary,
  analysis,
  onReviewNext,
  disabled = false,
}) => {
  if (!summary || !analysis) return null;

  const exportUrl = getParcelReviewExportUrl(analysis.analysis_id);

  return (
    <div className="bg-gis-card rounded-lg border border-gis-border p-4 shadow-lg space-y-3.5 text-xs">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gis-border pb-2">
        <h2 className="text-xs font-bold uppercase tracking-wider text-white flex items-center space-x-1.5">
          <ClipboardCheck className="w-3.5 h-3.5 text-gis-cyan" />
          <span>Parcel Review Queue</span>
        </h2>
        <span className="text-[11px] font-mono font-bold px-2 py-0.5 rounded bg-gis-surface text-gis-cyan border border-gis-border">
          {summary.total} Total
        </span>
      </div>

      {/* Breakdown Grid */}
      <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
        {/* Unreviewed */}
        <div className="p-2 rounded bg-gis-surface/50 border border-gis-border/70 flex items-center justify-between">
          <div className="flex items-center space-x-1.5 text-gis-textMuted">
            <Clock className="w-3 h-3 text-gray-400" />
            <span className="text-[10px] uppercase font-semibold">Unreviewed</span>
          </div>
          <span className="font-bold text-white text-sm">{summary.unreviewed}</span>
        </div>

        {/* Needs Review */}
        <div className="p-2 rounded bg-amber-500/10 border border-amber-500/30 flex items-center justify-between">
          <div className="flex items-center space-x-1.5 text-amber-300">
            <AlertTriangle className="w-3 h-3 text-amber-400" />
            <span className="text-[10px] uppercase font-semibold">Needs Review</span>
          </div>
          <span className="font-bold text-amber-400 text-sm">{summary.needs_review}</span>
        </div>

        {/* Accepted */}
        <div className="p-2 rounded bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-between">
          <div className="flex items-center space-x-1.5 text-emerald-300">
            <CheckCircle className="w-3 h-3 text-emerald-400" />
            <span className="text-[10px] uppercase font-semibold">Accepted</span>
          </div>
          <span className="font-bold text-emerald-400 text-sm">{summary.accepted}</span>
        </div>

        {/* Edited */}
        <div className="p-2 rounded bg-blue-500/10 border border-blue-500/30 flex items-center justify-between">
          <div className="flex items-center space-x-1.5 text-blue-300">
            <Edit3 className="w-3 h-3 text-blue-400" />
            <span className="text-[10px] uppercase font-semibold">Edited</span>
          </div>
          <span className="font-bold text-blue-400 text-sm">{summary.edited}</span>
        </div>

        {/* Rejected */}
        <div className="p-2 rounded bg-red-500/10 border border-red-500/30 col-span-2 flex items-center justify-between">
          <div className="flex items-center space-x-1.5 text-red-300">
            <XCircle className="w-3 h-3 text-red-400" />
            <span className="text-[10px] uppercase font-semibold">Rejected</span>
          </div>
          <span className="font-bold text-red-400 text-sm">{summary.rejected}</span>
        </div>
      </div>

      {/* Review Next Priority Button (Step 9) */}
      <button
        type="button"
        disabled={disabled || summary.total === 0}
        onClick={onReviewNext}
        className="w-full py-2.5 px-3 bg-gradient-to-r from-gis-cyan/90 to-emerald-500 hover:from-gis-cyan hover:to-emerald-400 text-gis-bg font-extrabold rounded shadow-md transition-all flex items-center justify-center space-x-2 text-xs uppercase tracking-wider group"
      >
        <span>Review Next Candidate</span>
        <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
      </button>

      {/* Export Reviewed GIS Data (Step 12) */}
      <a
        href={exportUrl}
        download={`reviewed_parcels_${analysis.analysis_id}.geojson`}
        className="w-full py-2 px-3 bg-gis-surface hover:bg-gis-surface/80 border border-gis-border text-white rounded text-xs font-semibold transition-colors flex items-center justify-center space-x-2"
      >
        <Download className="w-3.5 h-3.5 text-emerald-400" />
        <span>Export Reviewed GIS-Ready GeoJSON</span>
      </a>
    </div>
  );
};

export default ReviewQueuePanel;
