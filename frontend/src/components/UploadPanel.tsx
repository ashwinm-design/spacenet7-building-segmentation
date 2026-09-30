/**
 * frontend/src/components/UploadPanel.tsx
 *
 * Drag-and-drop file upload panel supporting GeoTIFF, TIFF, PNG, and JPEG imagery,
 * with optional ROI subregion input and real-time validation.
 */

import React, { useState, useRef } from 'react';
import { UploadCloud, FileSpreadsheet, Image as ImageIcon, Crosshair, ChevronDown, ChevronUp, AlertTriangle } from 'lucide-react';
import { formatFileSize } from '../utils/formatting';

interface UploadPanelProps {
  onAnalyze: (file: File, roi?: string) => void;
  isProcessing: boolean;
  disabled: boolean;
}

const ALLOWED_EXTENSIONS = ['.tif', '.tiff', '.png', '.jpg', '.jpeg'];

export const UploadPanel: React.FC<UploadPanelProps> = ({
  onAnalyze,
  isProcessing,
  disabled,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [roi, setRoi] = useState('');
  const [showRoiOptions, setShowRoiOptions] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndSetFile = (file: File) => {
    setErrorMsg(null);
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setErrorMsg(`Unsupported file type '${ext}'. Please upload GeoTIFF, TIFF, PNG, or JPEG.`);
      return;
    }
    if (file.size === 0) {
      setErrorMsg('Selected file is empty (0 bytes).');
      return;
    }
    setSelectedFile(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (disabled || isProcessing) return;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile || isProcessing || disabled) return;
    onAnalyze(selectedFile, roi.trim() || undefined);
  };

  return (
    <div className="bg-gis-card rounded-lg border border-gis-border p-4 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-gis-textMuted flex items-center space-x-1.5">
          <UploadCloud className="w-3.5 h-3.5 text-gis-accent" />
          <span>Imagery Upload</span>
        </h2>
        <span className="text-[10px] text-gis-textMuted">Max 250 MB</span>
      </div>

      {/* Drag & Drop Area */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          if (!isProcessing && !disabled) setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => {
          if (!isProcessing && !disabled) fileInputRef.current?.click();
        }}
        className={`relative border-2 border-dashed rounded-lg p-5 text-center cursor-pointer transition-colors ${
          dragOver
            ? 'border-gis-accent bg-gis-surface/80'
            : selectedFile
            ? 'border-gis-cyan/50 bg-gis-surface/40'
            : 'border-gis-border hover:border-slate-500 bg-gis-surface/30'
        } ${isProcessing || disabled ? 'opacity-60 cursor-not-allowed' : ''}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".tif,.tiff,.png,.jpg,.jpeg"
          onChange={handleFileInput}
          className="hidden"
          disabled={isProcessing || disabled}
        />

        {selectedFile ? (
          <div className="space-y-1">
            <div className="w-8 h-8 mx-auto rounded-full bg-gis-cyan/20 text-gis-cyan flex items-center justify-center">
              <ImageIcon className="w-4 h-4" />
            </div>
            <p className="text-xs font-semibold text-white truncate max-w-[240px] mx-auto">
              {selectedFile.name}
            </p>
            <p className="text-[11px] text-gis-textMuted">
              {formatFileSize(selectedFile.size)} • {selectedFile.type || 'Raster File'}
            </p>
            <p className="text-[10px] text-gis-accent">Click or drop to replace</p>
          </div>
        ) : (
          <div className="space-y-1.5">
            <div className="w-8 h-8 mx-auto rounded-full bg-gis-surface text-gis-textMuted flex items-center justify-center">
              <UploadCloud className="w-4 h-4" />
            </div>
            <p className="text-xs text-gis-textPrimary font-medium">
              Drop aerial or drone imagery here
            </p>
            <p className="text-[10px] text-gis-textMuted">
              GeoTIFF, TIFF, PNG, or JPEG
            </p>
          </div>
        )}
      </div>

      {errorMsg && (
        <div className="mt-2.5 p-2 rounded bg-gis-danger/10 border border-gis-danger/30 text-gis-danger text-xs flex items-center space-x-2">
          <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Advanced ROI Accordion */}
      <div className="mt-3">
        <button
          type="button"
          onClick={() => setShowRoiOptions(!showRoiOptions)}
          className="w-full flex items-center justify-between text-xs text-gis-textMuted hover:text-white py-1"
        >
          <span className="flex items-center space-x-1">
            <Crosshair className="w-3 h-3 text-gis-cyan" />
            <span>Region of Interest (ROI)</span>
          </span>
          {showRoiOptions ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>

        {showRoiOptions && (
          <div className="mt-2 p-2.5 rounded bg-gis-surface/50 border border-gis-border space-y-1.5">
            <label className="text-[11px] text-gis-textMuted block">
              Bounding Box (xmin,ymin,xmax,ymax)
            </label>
            <input
              type="text"
              placeholder="e.g. 3072,2048,4096,3072"
              value={roi}
              onChange={(e) => setRoi(e.target.value)}
              disabled={isProcessing || disabled}
              className="w-full px-2.5 py-1 text-xs bg-gis-bg rounded border border-gis-border text-white focus:outline-none focus:border-gis-accent font-mono"
            />
            <p className="text-[10px] text-gis-textMuted">
              Optional pixel crop for large orthomosaics.
            </p>
          </div>
        )}
      </div>

      {/* Analyze Button */}
      <button
        onClick={handleSubmit}
        disabled={!selectedFile || isProcessing || disabled}
        className="mt-4 w-full py-2.5 px-4 rounded-md font-medium text-xs text-white bg-gradient-to-r from-gis-accent to-gis-cyan hover:from-sky-500 hover:to-cyan-400 focus:outline-none shadow-md shadow-sky-950/40 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center space-x-2"
      >
        <span>{isProcessing ? 'Processing Imagery...' : 'Analyze Image'}</span>
      </button>
    </div>
  );
};

export default UploadPanel;
