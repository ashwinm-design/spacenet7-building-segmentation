/**
 * frontend/src/components/Header.tsx
 *
 * Header component displaying title, subtitle, backend connectivity status,
 * and quick-access Demo trigger.
 */

import React from 'react';
import { Layers, Activity, Play, CheckCircle2, AlertCircle } from 'lucide-react';

interface HeaderProps {
  backendConnected: boolean;
  onTryDemo: () => void;
  isProcessing: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  backendConnected,
  onTryDemo,
  isProcessing,
}) => {
  return (
    <header className="h-16 bg-gis-card border-b border-gis-border px-6 flex items-center justify-between flex-shrink-0 z-30">
      {/* Brand & Subtitle */}
      <div className="flex items-center space-x-3">
        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-gis-accent to-gis-cyan flex items-center justify-center text-white shadow-lg shadow-cyan-950/30">
          <Layers className="w-5 h-5" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-base font-bold tracking-tight text-white">
              AI-Assisted Urban Parcel Mapping
            </h1>
            <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-gis-surface text-gis-accent border border-gis-accent/30">
              SIH26012
            </span>
          </div>
          <p className="text-xs text-gis-textMuted hidden sm:block">
            High-Resolution Building Segmentation & Cadastral Feature Extraction
          </p>
        </div>
      </div>

      {/* 5-Step Workflow Indicator (Section 11) */}
      <div className="hidden lg:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-gis-surface/60 border border-gis-border text-[11px] font-mono text-gis-textMuted">
        <span className="text-gis-cyan font-semibold">1. Upload</span>
        <span className="text-slate-600">→</span>
        <span className="text-gis-accent font-semibold">2. AI Analysis</span>
        <span className="text-slate-600">→</span>
        <span className="text-emerald-400 font-semibold">3. Review Map</span>
        <span className="text-slate-600">→</span>
        <span className="text-amber-400 font-semibold">4. Verify Parcels</span>
        <span className="text-slate-600">→</span>
        <span className="text-purple-400 font-semibold">5. Export GIS</span>
      </div>

      {/* Actions & Status */}
      <div className="flex items-center space-x-4">
        {/* Try Demo Button */}
        <button
          onClick={onTryDemo}
          disabled={!backendConnected || isProcessing}
          className="flex items-center space-x-1.5 px-3 py-1.5 text-xs font-medium rounded-md bg-gis-surface hover:bg-gis-border text-white border border-gis-border transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          title="Run inference on Open Cities GeoTIFF subregion (254 buildings)"
        >
          <Play className="w-3.5 h-3.5 text-gis-cyan fill-gis-cyan" />
          <span>Try Demo</span>
        </button>

        {/* Backend Status Indicator */}
        <div className="flex items-center space-x-2 px-3 py-1 rounded-full bg-gis-surface/80 border border-gis-border text-xs">
          <span className="text-gis-textMuted">Backend</span>
          {backendConnected ? (
            <span className="flex items-center space-x-1.5 text-gis-success font-medium">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-gis-success opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-gis-success"></span>
              </span>
              <span>Connected</span>
            </span>
          ) : (
            <span className="flex items-center space-x-1.5 text-gis-danger font-medium">
              <span className="inline-flex rounded-full h-2 w-2 bg-gis-danger"></span>
              <span>Offline</span>
            </span>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;
