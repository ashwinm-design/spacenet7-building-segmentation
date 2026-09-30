/**
 * frontend/src/components/ParcelInspectionPanel.tsx
 *
 * Dedicated Parcel Intelligence & Human Verification Panel (Phase 14B).
 * Displays AI candidate confidence, multi-criteria evidence components,
 * spatial metrics, uncertainty flags, vertex editing controls, and reviewer actions.
 */

import React, { useState, useEffect } from 'react';
import {
  X,
  CheckCircle,
  AlertTriangle,
  RotateCcw,
  Edit3,
  XCircle,
  Clock,
  ShieldAlert,
  Building,
  Navigation,
  Shapes,
  Maximize2,
  FileText,
  Save,
  Eye,
  Layers,
} from 'lucide-react';
import { ParcelReviewItem, ReviewStatus } from '../types/analysis';
import { formatArea, formatConfidence } from '../utils/formatting';

interface ParcelInspectionPanelProps {
  review: ParcelReviewItem | null;
  onClose: () => void;
  onUpdateStatus: (status: ReviewStatus, note?: string) => Promise<void>;
  isEditingGeometry: boolean;
  onStartEditGeometry: () => void;
  onCancelEditGeometry: () => void;
  onResetGeometry: () => Promise<void>;
  showOriginalGeometry: boolean;
  onToggleShowOriginal: (show: boolean) => void;
}

export const ParcelInspectionPanel: React.FC<ParcelInspectionPanelProps> = ({
  review,
  onClose,
  onUpdateStatus,
  isEditingGeometry,
  onStartEditGeometry,
  onCancelEditGeometry,
  onResetGeometry,
  showOriginalGeometry,
  onToggleShowOriginal,
}) => {
  const [note, setNote] = useState<string>('');
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

  useEffect(() => {
    if (review) {
      setNote(review.reviewer_note || '');
    }
  }, [review]);

  if (!review) return null;

  const candidateIdStr = `P${String(review.candidate_id).padStart(3, '0')}`;

  const statusConfig: Record<
    ReviewStatus,
    { label: string; badgeClass: string; icon: React.ReactNode }
  > = {
    unreviewed: {
      label: 'UNREVIEWED',
      badgeClass: 'bg-gray-700/80 text-gray-300 border-gray-600',
      icon: <Clock className="w-3.5 h-3.5 text-gray-400" />,
    },
    needs_review: {
      label: 'NEEDS REVIEW',
      badgeClass: 'bg-amber-500/20 text-amber-400 border-amber-500/40',
      icon: <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />,
    },
    accepted: {
      label: 'ACCEPTED',
      badgeClass: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40',
      icon: <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />,
    },
    edited: {
      label: 'EDITED',
      badgeClass: 'bg-blue-500/20 text-blue-400 border-blue-500/40',
      icon: <Edit3 className="w-3.5 h-3.5 text-blue-400" />,
    },
    rejected: {
      label: 'REJECTED',
      badgeClass: 'bg-red-500/20 text-red-400 border-red-500/40',
      icon: <XCircle className="w-3.5 h-3.5 text-red-400" />,
    },
  };

  const currentStatusInfo = statusConfig[review.review_status] || statusConfig.unreviewed;

  const bandColor =
    review.confidence_band === 'HIGH'
      ? 'text-emerald-400'
      : review.confidence_band === 'MODERATE'
      ? 'text-amber-400'
      : 'text-rose-400';

  // Parse uncertainty flags
  const flagsList: string[] = Array.isArray(review.flags)
    ? review.flags
    : typeof review.flags === 'string' && review.flags.trim()
    ? review.flags.split(',').map((f) => f.trim()).filter(Boolean)
    : [];

  const handleAction = async (status: ReviewStatus) => {
    setIsSubmitting(true);
    try {
      await onUpdateStatus(status, note);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleSaveNoteOnly = async () => {
    setIsSubmitting(true);
    try {
      await onUpdateStatus(review.review_status, note);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="bg-gis-card rounded-lg border border-gis-border p-4 shadow-xl space-y-3.5 text-xs animate-in fade-in slide-in-from-top-2 duration-150 max-h-[85vh] overflow-y-auto">
      {/* 1. Header with Candidate ID & Status Badge */}
      <div className="flex items-center justify-between border-b border-gis-border pb-2.5">
        <div className="flex items-center space-x-2.5">
          <div className="w-7 h-7 rounded bg-gis-cyan/20 text-gis-cyan flex items-center justify-center font-mono text-xs font-bold border border-gis-cyan/30">
            {candidateIdStr}
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h3 className="font-bold text-white text-sm tracking-wide font-mono">
                PARCEL {candidateIdStr}
              </h3>
              <span
                className={`px-2 py-0.5 rounded text-[10px] font-bold border flex items-center space-x-1 ${currentStatusInfo.badgeClass}`}
              >
                {currentStatusInfo.icon}
                <span>{currentStatusInfo.label}</span>
              </span>
            </div>
            <span className="text-[10px] text-gis-textMuted font-mono">
              Preliminary Hybrid-Constrained AI Candidate
            </span>
          </div>
        </div>

        <button
          onClick={onClose}
          className="text-gis-textMuted hover:text-white p-1 rounded-md hover:bg-gis-surface transition-colors"
          title="Close parcel inspection"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* 2. Confidence & Band Bar */}
      <div className="grid grid-cols-2 gap-2 bg-gis-surface/60 rounded-lg p-2.5 border border-gis-border/80">
        <div>
          <span className="text-[10px] text-gis-textMuted uppercase font-semibold tracking-wider block">
            Confidence
          </span>
          <span className={`text-lg font-mono font-bold ${bandColor}`}>
            {(review.confidence * 100).toFixed(2)}%
          </span>
        </div>
        <div>
          <span className="text-[10px] text-gis-textMuted uppercase font-semibold tracking-wider block">
            Confidence Band
          </span>
          <span className={`text-sm font-bold tracking-wider uppercase font-mono ${bandColor}`}>
            {review.confidence_band}
          </span>
        </div>
      </div>

      {/* 3. Evidence Breakdown Section */}
      <div className="space-y-1.5 bg-gis-surface/40 rounded-lg p-2.5 border border-gis-border/60">
        <div className="flex items-center justify-between border-b border-gis-border/60 pb-1">
          <span className="text-[10px] font-bold uppercase text-gis-cyan tracking-wider flex items-center space-x-1">
            <Shapes className="w-3.5 h-3.5" />
            <span>Evidence Dimensions</span>
          </span>
          <span className="text-[9px] text-gis-textMuted font-mono">Multi-Criteria Fusion</span>
        </div>

        <div className="space-y-2 pt-1 font-mono text-[11px]">
          {/* Building Evidence */}
          <div>
            <div className="flex justify-between items-center text-gis-textPrimary mb-0.5">
              <span className="text-[10px] text-gis-textMuted flex items-center space-x-1">
                <Building className="w-3 h-3 text-amber-400 mr-1" />
                <span>Building Evidence</span>
              </span>
              <span className="font-bold text-amber-300">
                {(review.conf_building_evidence ?? 0.68).toFixed(2)}
              </span>
            </div>
            <div className="w-full bg-gis-bg rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-amber-400 h-1.5 rounded-full"
                style={{ width: `${Math.min(100, (review.conf_building_evidence ?? 0.68) * 100)}%` }}
              />
            </div>
          </div>

          {/* Road Access */}
          <div>
            <div className="flex justify-between items-center text-gis-textPrimary mb-0.5">
              <span className="text-[10px] text-gis-textMuted flex items-center space-x-1">
                <Navigation className="w-3 h-3 text-cyan-400 mr-1" />
                <span>Road Access</span>
              </span>
              <span className="font-bold text-cyan-300">
                {(review.conf_road_access ?? 0.95).toFixed(2)}
              </span>
            </div>
            <div className="w-full bg-gis-bg rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-cyan-400 h-1.5 rounded-full"
                style={{ width: `${Math.min(100, (review.conf_road_access ?? 0.95) * 100)}%` }}
              />
            </div>
          </div>

          {/* Geometry Quality */}
          <div>
            <div className="flex justify-between items-center text-gis-textPrimary mb-0.5">
              <span className="text-[10px] text-gis-textMuted flex items-center space-x-1">
                <Maximize2 className="w-3 h-3 text-emerald-400 mr-1" />
                <span>Geometry Quality</span>
              </span>
              <span className="font-bold text-emerald-300">
                {(review.conf_geometry_quality ?? 1.0).toFixed(2)}
              </span>
            </div>
            <div className="w-full bg-gis-bg rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-emerald-400 h-1.5 rounded-full"
                style={{ width: `${Math.min(100, (review.conf_geometry_quality ?? 1.0) * 100)}%` }}
              />
            </div>
          </div>

          {/* Topology Quality */}
          <div>
            <div className="flex justify-between items-center text-gis-textPrimary mb-0.5">
              <span className="text-[10px] text-gis-textMuted flex items-center space-x-1">
                <CheckCircle className="w-3 h-3 text-blue-400 mr-1" />
                <span>Topology Quality</span>
              </span>
              <span className="font-bold text-blue-300">
                {(review.conf_topological_plausibility ?? 1.0).toFixed(2)}
              </span>
            </div>
            <div className="w-full bg-gis-bg rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-blue-400 h-1.5 rounded-full"
                style={{ width: `${Math.min(100, (review.conf_topological_plausibility ?? 1.0) * 100)}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* 4. Cadastral Spatial Metrics */}
      <div className="space-y-1.5 bg-gis-surface/40 rounded-lg p-2.5 border border-gis-border/60">
        <span className="text-[10px] font-bold uppercase text-gis-cyan tracking-wider block border-b border-gis-border/60 pb-1">
          Spatial Metrics
        </span>
        <div className="grid grid-cols-2 gap-2 text-xs font-mono pt-1">
          <div className="p-1.5 rounded bg-gis-bg/60 border border-gis-border/50">
            <span className="text-[9px] text-gis-textMuted uppercase block">Area</span>
            <span className="text-white font-bold">
              {review.area_m2 !== undefined ? `${review.area_m2.toFixed(1)} m²` : 'N/A'}
            </span>
          </div>

          <div className="p-1.5 rounded bg-gis-bg/60 border border-gis-border/50">
            <span className="text-[9px] text-gis-textMuted uppercase block">Building Coverage</span>
            <span className="text-white font-bold">
              {review.building_coverage !== undefined
                ? `${(review.building_coverage * 100).toFixed(1)}%`
                : 'N/A'}
            </span>
          </div>

          <div className="p-1.5 rounded bg-gis-bg/60 border border-gis-border/50">
            <span className="text-[9px] text-gis-textMuted uppercase block">Road Frontage</span>
            <span className="text-cyan-400 font-bold">
              {review.road_frontage !== undefined ? `${review.road_frontage.toFixed(1)} m` : '0.0 m'}
            </span>
          </div>

          <div className="p-1.5 rounded bg-gis-bg/60 border border-gis-border/50">
            <span className="text-[9px] text-gis-textMuted uppercase block">Nearest Road</span>
            <span className="text-blue-300 font-bold">
              {review.nearest_road_distance !== undefined
                ? `${review.nearest_road_distance.toFixed(1)} m`
                : '0.0 m'}
            </span>
          </div>
        </div>
      </div>

      {/* 5. Uncertainty Flags */}
      <div className="space-y-1 bg-gis-surface/40 rounded-lg p-2.5 border border-gis-border/60">
        <span className="text-[10px] font-bold uppercase text-gis-cyan tracking-wider flex items-center space-x-1 border-b border-gis-border/60 pb-1">
          <AlertTriangle className="w-3.5 h-3.5 text-yellow-400" />
          <span>Uncertainty Flags ({flagsList.length})</span>
        </span>
        <div className="pt-1">
          {flagsList.length > 0 ? (
            <div className="space-y-1">
              {flagsList.map((flag, idx) => (
                <div
                  key={idx}
                  className="p-1.5 rounded bg-yellow-500/10 border border-yellow-500/30 text-[10px] text-yellow-300 font-mono flex items-center space-x-1.5"
                >
                  <AlertTriangle className="w-3 h-3 text-yellow-400 shrink-0" />
                  <span className="font-semibold">{flag.replace(/_/g, ' ')}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="p-1.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-[10px] text-emerald-300 font-mono flex items-center space-x-1.5">
              <CheckCircle className="w-3 h-3 text-emerald-400 shrink-0" />
              <span>No uncertainty flags raised</span>
            </div>
          )}
        </div>
      </div>

      {/* 6. Original vs Current Geometry Comparison (Step 13) */}
      {review.geometry_modified && (
        <div className="p-2.5 rounded bg-blue-500/10 border border-blue-500/30 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-blue-300 uppercase flex items-center space-x-1">
              <Layers className="w-3 h-3" />
              <span>Human Geometry Modification</span>
            </span>
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-blue-900/60 text-blue-300 font-mono">
              Edited
            </span>
          </div>

          <div className="grid grid-cols-2 gap-1.5">
            <button
              type="button"
              onClick={() => onToggleShowOriginal(false)}
              className={`py-1 px-2 rounded text-[10px] font-bold border transition-colors flex items-center justify-center space-x-1 ${
                !showOriginalGeometry
                  ? 'bg-blue-600 text-white border-blue-500'
                  : 'bg-gis-surface/80 text-gis-textMuted border-gis-border hover:text-white'
              }`}
            >
              <Eye className="w-3 h-3" />
              <span>Show Current</span>
            </button>
            <button
              type="button"
              onClick={() => onToggleShowOriginal(true)}
              className={`py-1 px-2 rounded text-[10px] font-bold border transition-colors flex items-center justify-center space-x-1 ${
                showOriginalGeometry
                  ? 'bg-amber-600 text-white border-amber-500'
                  : 'bg-gis-surface/80 text-gis-textMuted border-gis-border hover:text-white'
              }`}
            >
              <RotateCcw className="w-3 h-3" />
              <span>Show Original AI</span>
            </button>
          </div>

          <button
            type="button"
            onClick={onResetGeometry}
            className="w-full py-1 bg-gis-surface hover:bg-gis-surface/80 border border-gis-border text-gis-textMuted hover:text-white rounded text-[10px] font-medium transition-colors flex items-center justify-center space-x-1"
          >
            <RotateCcw className="w-3 h-3 text-amber-400" />
            <span>Reset to Original AI Geometry</span>
          </button>
        </div>
      )}

      {/* 7. Review Actions (Step 6, Step 10) */}
      <div className="space-y-2 pt-1 border-t border-gis-border">
        <span className="text-[10px] font-bold uppercase text-gis-textMuted tracking-wider block">
          Review Decision
        </span>

        <div className="grid grid-cols-2 gap-1.5">
          {/* ACCEPT Button */}
          <button
            type="button"
            disabled={isSubmitting}
            onClick={() => handleAction('accepted')}
            className={`py-2 px-2.5 rounded font-bold text-xs flex items-center justify-center space-x-1.5 transition-all shadow-sm ${
              review.review_status === 'accepted'
                ? 'bg-emerald-500 text-gis-bg ring-2 ring-emerald-400 font-extrabold'
                : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 hover:bg-emerald-500/30'
            }`}
          >
            <CheckCircle className="w-3.5 h-3.5" />
            <span>ACCEPT</span>
          </button>

          {/* EDIT GEOMETRY Button */}
          <button
            type="button"
            onClick={isEditingGeometry ? onCancelEditGeometry : onStartEditGeometry}
            className={`py-2 px-2.5 rounded font-bold text-xs flex items-center justify-center space-x-1.5 transition-all shadow-sm ${
              isEditingGeometry
                ? 'bg-blue-500 text-white ring-2 ring-blue-400 animate-pulse font-extrabold'
                : review.review_status === 'edited'
                ? 'bg-blue-500/30 text-blue-300 border border-blue-400'
                : 'bg-blue-500/20 text-blue-400 border border-blue-500/40 hover:bg-blue-500/30'
            }`}
          >
            <Edit3 className="w-3.5 h-3.5" />
            <span>{isEditingGeometry ? 'DONE EDIT' : 'EDIT GEOM'}</span>
          </button>

          {/* NEEDS REVIEW Button */}
          <button
            type="button"
            disabled={isSubmitting}
            onClick={() => handleAction('needs_review')}
            className={`py-2 px-2.5 rounded font-bold text-xs flex items-center justify-center space-x-1.5 transition-all shadow-sm ${
              review.review_status === 'needs_review'
                ? 'bg-amber-500 text-gis-bg ring-2 ring-amber-400 font-extrabold'
                : 'bg-amber-500/20 text-amber-400 border border-amber-500/40 hover:bg-amber-500/30'
            }`}
          >
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>NEEDS REVIEW</span>
          </button>

          {/* REJECT Button */}
          <button
            type="button"
            disabled={isSubmitting}
            onClick={() => handleAction('rejected')}
            className={`py-2 px-2.5 rounded font-bold text-xs flex items-center justify-center space-x-1.5 transition-all shadow-sm ${
              review.review_status === 'rejected'
                ? 'bg-red-500 text-white ring-2 ring-red-400 font-extrabold'
                : 'bg-red-500/20 text-red-400 border border-red-500/40 hover:bg-red-500/30'
            }`}
          >
            <XCircle className="w-3.5 h-3.5" />
            <span>REJECT</span>
          </button>
        </div>

        {/* Reviewer Note */}
        <div className="space-y-1 pt-1">
          <label className="text-[10px] text-gis-textMuted uppercase font-semibold flex items-center space-x-1">
            <FileText className="w-3 h-3 text-gis-cyan" />
            <span>Reviewer Note</span>
          </label>
          <textarea
            value={note}
            onChange={(e) => setNote(e.target.value)}
            rows={2}
            placeholder="Add reviewer observations, field notes, or boundary remarks..."
            className="w-full bg-gis-surface border border-gis-border rounded p-2 text-xs text-white focus:outline-none focus:border-gis-cyan resize-none font-sans"
          />
          <button
            type="button"
            disabled={isSubmitting}
            onClick={handleSaveNoteOnly}
            className="w-full py-1.5 bg-gis-surface hover:bg-gis-surface/80 border border-gis-border text-gis-cyan rounded text-[11px] font-bold transition-colors flex items-center justify-center space-x-1"
          >
            <Save className="w-3 h-3" />
            <span>SAVE REVIEW NOTE</span>
          </button>
        </div>
      </div>

      {/* 8. Mandatory Cadastral Legal Disclaimer (Step 16) */}
      <div className="p-2.5 rounded bg-amber-500/10 border border-amber-500/30 text-[10px] text-amber-300/90 flex items-start space-x-2 leading-relaxed">
        <ShieldAlert className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
        <p>
          <strong>Notice:</strong> AI-generated parcel boundaries are preliminary spatial candidates
          and require professional cadastral/survey verification. Review status does not establish
          legal ownership or boundary rights.
        </p>
      </div>
    </div>
  );
};

export default ParcelInspectionPanel;
