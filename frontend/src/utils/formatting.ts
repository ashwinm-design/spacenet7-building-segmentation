/**
 * frontend/src/utils/formatting.ts
 *
 * Number, unit, and string formatters for GIS measurements.
 */

export function formatArea(val: number | undefined | null, unit: string = 'm²'): string {
  if (val === undefined || val === null || isNaN(val)) return '—';
  const formatted = val.toLocaleString('en-US', {
    minimumFractionDigits: 1,
    maximumFractionDigits: 2,
  });
  return `${formatted} ${unit}`;
}

export function formatPerimeter(val: number | undefined | null, unit: string = 'm'): string {
  if (val === undefined || val === null || isNaN(val)) return '—';
  const formatted = val.toLocaleString('en-US', {
    minimumFractionDigits: 1,
    maximumFractionDigits: 2,
  });
  return `${formatted} ${unit}`;
}

export function formatConfidence(val: number | undefined | null): string {
  if (val === undefined || val === null || isNaN(val)) return '—';
  return (val * 100).toFixed(1) + '%';
}

export function formatNumber(val: number | undefined | null): string {
  if (val === undefined || val === null || isNaN(val)) return '—';
  return val.toLocaleString('en-US');
}

export function formatSeconds(sec: number | undefined | null): string {
  if (sec === undefined || sec === null || isNaN(sec)) return '—';
  return `${sec.toFixed(2)}s`;
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
