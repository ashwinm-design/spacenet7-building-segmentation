/**
 * frontend/src/components/BuildingPanel.tsx
 *
 * Detailed inspector card for a selected building instance with cadastral metrics.
 */

import React from 'react';
import { X, ShieldAlert, CheckCircle2, Crosshair, MapPin, Maximize2 } from 'lucide-react';
import { BuildingDetailResponse } from '../types/analysis';
import { formatArea, formatPerimeter, formatConfidence } from '../utils/formatting';

interface BuildingPanelProps {
  building: BuildingDetailResponse | null;
  onClose: () => void;
}

export const BuildingPanel: React.FC<BuildingPanelProps> = ({ building, onClose }) => {
  if (!building) return null;

  const areaUnit = building.area_unit === 'm^2' ? 'm²' : building.area_unit === 'pixels^2' ? 'px²' : building.area_unit;
  const perimUnit = building.perimeter_unit === 'meters' ? 'm' : building.perimeter_unit;

  return (
    <div className="bg-gis-card rounded-lg border border-gis-border p-4 shadow-lg space-y-3 animate-in fade-in slide-in-from-top-2 duration-150">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gis-border pb-2.5">
        <div className="flex items-center space-x-2">
          <div className="w-6 h-6 rounded bg-gis-cyan/20 text-gis-cyan flex items-center justify-center font-mono text-xs font-bold">
            #{building.id}
          </div>
          <div>
            <h3 className="text-xs font-bold text-white uppercase tracking-wider">
              Building Footprint #{building.id}
            </h3>
            <span className="text-[10px] text-gis-textMuted font-mono">
              {building.is_georeferenced ? 'Georeferenced Vector' : 'Pixel Geometry'}
            </span>
          </div>
        </div>

        <button
          onClick={onClose}
          className="text-gis-textMuted hover:text-white p-1 rounded-md hover:bg-gis-surface transition-colors"
          title="Close building details"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="p-2 rounded bg-gis-surface/50 border border-gis-border/60">
          <span className="text-[10px] text-gis-textMuted uppercase font-medium block">
            Footprint Area
          </span>
          <span className="text-sm font-bold text-white font-mono">
            {formatArea(building.area, areaUnit)}
          </span>
        </div>

        <div className="p-2 rounded bg-gis-surface/50 border border-gis-border/60">
          <span className="text-[10px] text-gis-textMuted uppercase font-medium block">
            Perimeter
          </span>
          <span className="text-sm font-bold text-white font-mono">
            {formatPerimeter(building.perimeter, perimUnit)}
          </span>
        </div>

        <div className="p-2 rounded bg-gis-surface/50 border border-gis-border/60">
          <span className="text-[10px] text-gis-textMuted uppercase font-medium block">
            Confidence (Mean)
          </span>
          <span className="text-sm font-bold text-gis-success font-mono">
            {formatConfidence(building.confidence)}
          </span>
        </div>

        <div className="p-2 rounded bg-gis-surface/50 border border-gis-border/60">
          <span className="text-[10px] text-gis-textMuted uppercase font-medium block">
            Confidence (Min)
          </span>
          <span className="text-sm font-bold text-gis-warning font-mono">
            {formatConfidence(building.min_confidence)}
          </span>
        </div>
      </div>

      {/* Centroid coordinates */}
      <div className="p-2 rounded bg-gis-surface/40 border border-gis-border/50 text-[11px] font-mono space-y-0.5">
        <div className="flex items-center space-x-1 text-gis-textMuted text-[10px] uppercase">
          <Crosshair className="w-3 h-3 text-gis-cyan" />
          <span>Centroid ({building.is_georeferenced ? 'Projected X/Y' : 'Pixel X/Y'})</span>
        </div>
        <div className="text-gis-textPrimary flex justify-between">
          <span>X: {building.centroid_x.toFixed(2)}</span>
          <span>Y: {building.centroid_y.toFixed(2)}</span>
        </div>
      </div>

      {/* Mandatory Cadastral Legal Disclaimer */}
      <div className="p-2.5 rounded bg-amber-500/10 border border-amber-500/30 text-[11px] text-amber-300/90 flex items-start space-x-2">
        <ShieldAlert className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
        <p className="leading-snug">
          <strong>AI-extracted building footprint.</strong> Not a legal property boundary or cadastral title.
        </p>
      </div>
    </div>
  );
};

export default BuildingPanel;
