/**
 * frontend/src/components/ExportPanel.tsx
 *
 * SIH26012 Phase 15 Export Center:
 * Provides direct GIS-ready exports for:
 * 1. Building Footprints — GeoJSON
 * 2. Hybrid Roads — GeoJSON
 * 3. Preliminary Parcels — GeoJSON
 * 4. Reviewed Parcels — GeoJSON
 * 5. Analysis Summary — JSON
 * 6. Review Audit — JSON
 */

import React from 'react';
import { Download, FileJson, FileSpreadsheet, ShieldCheck, Map, Network, FileText, CheckCircle2 } from 'lucide-react';
import { AnalysisResponse } from '../types/analysis';
import {
  getGeoJsonDownloadUrl,
  getStatisticsDownloadUrl,
  getOverlayDownloadUrl,
  getParcelReviewExportUrl,
  getSummaryDownloadUrl,
  getAuditDownloadUrl,
  getRoadsDownloadUrl,
  getParcelsDownloadUrl,
} from '../api/analysis';

interface ExportPanelProps {
  analysis: AnalysisResponse | null;
}

export const ExportPanel: React.FC<ExportPanelProps> = ({ analysis }) => {
  if (!analysis) return null;

  const id = analysis.analysis_id;

  return (
    <div className="bg-gis-card rounded-lg border border-gis-border p-4 shadow-sm space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-gis-textMuted flex items-center space-x-1.5">
          <Download className="w-3.5 h-3.5 text-gis-accent" />
          <span>Export Center</span>
        </h2>
        <span className="text-[10px] text-gis-textMuted font-mono">
          ID: {id}
        </span>
      </div>

      <div className="space-y-2">
        {/* 1. Building Footprints — GeoJSON */}
        <a
          href={getGeoJsonDownloadUrl(id)}
          download={`buildings_${id}.geojson`}
          className="w-full flex items-center justify-between p-2 rounded-md bg-gis-surface/60 hover:bg-gis-surface border border-gis-border text-xs text-gis-textPrimary transition-colors group"
        >
          <div className="flex items-center space-x-2.5">
            <FileJson className="w-4 h-4 text-gis-cyan group-hover:scale-110 transition-transform" />
            <div>
              <div className="font-medium text-white">Building Footprints</div>
              <div className="text-[10px] text-gis-textMuted">GeoJSON • {analysis.crs || 'EPSG:32732'}</div>
            </div>
          </div>
          <Download className="w-3.5 h-3.5 text-gis-textMuted group-hover:text-white" />
        </a>

        {/* 2. Hybrid Roads — GeoJSON */}
        <a
          href={getRoadsDownloadUrl(id, 'hybrid')}
          download={`roads_hybrid_${id}.geojson`}
          className="w-full flex items-center justify-between p-2 rounded-md bg-gis-surface/60 hover:bg-gis-surface border border-gis-border text-xs text-gis-textPrimary transition-colors group"
        >
          <div className="flex items-center space-x-2.5">
            <Network className="w-4 h-4 text-emerald-400 group-hover:scale-110 transition-transform" />
            <div>
              <div className="font-medium text-white">Hybrid Roads</div>
              <div className="text-[10px] text-gis-textMuted">GeoJSON • Fused Corridors</div>
            </div>
          </div>
          <Download className="w-3.5 h-3.5 text-gis-textMuted group-hover:text-white" />
        </a>

        {/* 3. Preliminary Parcels — GeoJSON */}
        <a
          href={getParcelsDownloadUrl(id, 'hybrid')}
          download={`candidate_parcels_${id}.geojson`}
          className="w-full flex items-center justify-between p-2 rounded-md bg-gis-surface/60 hover:bg-gis-surface border border-gis-border text-xs text-gis-textPrimary transition-colors group"
        >
          <div className="flex items-center space-x-2.5">
            <Map className="w-4 h-4 text-purple-400 group-hover:scale-110 transition-transform" />
            <div>
              <div className="font-medium text-white">Preliminary Parcels</div>
              <div className="text-[10px] text-gis-textMuted">GeoJSON • AI Hypotheses</div>
            </div>
          </div>
          <Download className="w-3.5 h-3.5 text-gis-textMuted group-hover:text-white" />
        </a>

        {/* 4. Reviewed Parcels — GeoJSON */}
        <a
          href={getParcelReviewExportUrl(id)}
          download={`reviewed_parcels_${id}.geojson`}
          className="w-full flex items-center justify-between p-2 rounded-md bg-gis-surface/60 hover:bg-gis-surface border border-emerald-500/40 text-xs text-gis-textPrimary transition-colors group"
        >
          <div className="flex items-center space-x-2.5">
            <ShieldCheck className="w-4 h-4 text-gis-success group-hover:scale-110 transition-transform" />
            <div>
              <div className="font-medium text-white">Reviewed Parcels</div>
              <div className="text-[10px] text-emerald-400">GIS-Ready • Audit Attributes</div>
            </div>
          </div>
          <Download className="w-3.5 h-3.5 text-gis-textMuted group-hover:text-white" />
        </a>

        {/* 5. Analysis Summary — JSON */}
        <a
          href={getSummaryDownloadUrl(id)}
          download={`analysis_summary_${id}.json`}
          className="w-full flex items-center justify-between p-2 rounded-md bg-gis-surface/60 hover:bg-gis-surface border border-gis-border text-xs text-gis-textPrimary transition-colors group"
        >
          <div className="flex items-center space-x-2.5">
            <FileText className="w-4 h-4 text-amber-400 group-hover:scale-110 transition-transform" />
            <div>
              <div className="font-medium text-white">Analysis Summary</div>
              <div className="text-[10px] text-gis-textMuted">JSON • Timings & KPI Metrics</div>
            </div>
          </div>
          <Download className="w-3.5 h-3.5 text-gis-textMuted group-hover:text-white" />
        </a>

        {/* 6. Review Audit — JSON */}
        <a
          href={getAuditDownloadUrl(id)}
          download={`parcel_reviews_audit_${id}.json`}
          className="w-full flex items-center justify-between p-2 rounded-md bg-gis-surface/60 hover:bg-gis-surface border border-gis-border text-xs text-gis-textPrimary transition-colors group"
        >
          <div className="flex items-center space-x-2.5">
            <CheckCircle2 className="w-4 h-4 text-sky-400 group-hover:scale-110 transition-transform" />
            <div>
              <div className="font-medium text-white">Review Audit Trail</div>
              <div className="text-[10px] text-gis-textMuted">JSON • Immutable Action Log</div>
            </div>
          </div>
          <Download className="w-3.5 h-3.5 text-gis-textMuted group-hover:text-white" />
        </a>
      </div>
    </div>
  );
};

export default ExportPanel;
