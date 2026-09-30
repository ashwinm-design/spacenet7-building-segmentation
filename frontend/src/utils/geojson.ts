/**
 * frontend/src/utils/geojson.ts
 *
 * Client-side projection engine and GeoJSON reprojector using proj4.
 * Reprojects projected coordinates (e.g. EPSG:32732) to WGS84 EPSG:4326
 * exclusively for MapLibre Web Mercator rendering.
 * Original downloaded GeoJSON remains untouched!
 */

import proj4 from 'proj4';
import { GeoJSONFeatureCollection, GeoJSONFeature } from '../types/analysis';

// Pre-define standard definitions
proj4.defs('EPSG:4326', '+proj=longlat +datum=WGS84 +no_defs');
proj4.defs('EPSG:3857', '+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs');
proj4.defs('EPSG:32732', '+proj=utm +zone=32 +south +datum=WGS84 +units=m +no_defs');
proj4.defs('EPSG:32632', '+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs');

/**
 * Ensures proj4 has a definition for the given EPSG code.
 * Supports standard UTM zones 1-60 North (EPSG:32601-32660) and South (EPSG:32701-32760).
 */
export function ensureProjectionDef(crsStr: string): string {
  let epsgMatch = crsStr.match(/32[67]\d{2}/);
  if (!epsgMatch) {
    const generalMatch = crsStr.match(/EPSG:(\d+)/i);
    if (generalMatch) {
      const code = generalMatch[1];
      if ((proj4.defs as any)[`EPSG:${code}`]) return `EPSG:${code}`;
    }
  }

  if (epsgMatch) {
    const code = parseInt(epsgMatch[0], 10);
    const epsgKey = `EPSG:${code}`;
    if (!(proj4.defs as any)[epsgKey]) {
      const isSouth = code >= 32700;
      const zone = code % 100;
      const def = `+proj=utm +zone=${zone} ${isSouth ? '+south ' : ''}+datum=WGS84 +units=m +no_defs`;
      proj4.defs(epsgKey, def);
    }
    return epsgKey;
  }

  return 'EPSG:4326';
}

/**
 * Recursively reprojects coordinate arrays from source CRS to EPSG:4326 (lon, lat).
 */
function reprojectCoords(coords: any, sourceCrs: string): any {
  if (typeof coords[0] === 'number') {
    const [x, y] = coords;
    try {
      const [lon, lat] = proj4(sourceCrs, 'EPSG:4326', [x, y]);
      return [lon, lat];
    } catch {
      return [x, y];
    }
  }
  return coords.map((c: any) => reprojectCoords(c, sourceCrs));
}

/**
 * Creates a reprojected deep copy of a GeoJSON FeatureCollection for MapLibre display.
 */
export function reprojectGeoJSONForDisplay(
  geojson: GeoJSONFeatureCollection,
  crsStr: string | null
): GeoJSONFeatureCollection {
  if (!crsStr || crsStr.toUpperCase().includes('4326') || crsStr.toUpperCase().includes('WGS84')) {
    return JSON.parse(JSON.stringify(geojson));
  }

  const sourceCrs = ensureProjectionDef(crsStr);
  const cloned: GeoJSONFeatureCollection = JSON.parse(JSON.stringify(geojson));

  cloned.features = cloned.features.map((feature: GeoJSONFeature) => {
    if (feature.geometry && feature.geometry.coordinates) {
      feature.geometry.coordinates = reprojectCoords(
        feature.geometry.coordinates,
        sourceCrs
      );
    }
    return feature;
  });

  return cloned;
}

/**
 * Creates a reprojected deep copy of a single GeoJSON Geometry for MapLibre display.
 */
export function reprojectGeometryForDisplay(
  geom: any,
  crsStr: string | null
): any {
  if (!geom || !geom.coordinates) return geom;
  if (!crsStr || crsStr.toUpperCase().includes('4326') || crsStr.toUpperCase().includes('WGS84')) {
    return JSON.parse(JSON.stringify(geom));
  }

  const sourceCrs = ensureProjectionDef(crsStr);
  const cloned = JSON.parse(JSON.stringify(geom));
  cloned.coordinates = reprojectCoords(cloned.coordinates, sourceCrs);
  return cloned;
}

export interface BoundingBox {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/**
 * Calculates geographic or pixel bounding box from a FeatureCollection.
 */
export function calculateBounds(geojson: GeoJSONFeatureCollection): BoundingBox | null {
  if (!geojson.features || geojson.features.length === 0) return null;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  function traverse(coords: any) {
    if (typeof coords[0] === 'number') {
      const [x, y] = coords;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      return;
    }
    coords.forEach(traverse);
  }

  for (const f of geojson.features) {
    if (f.geometry && f.geometry.coordinates) {
      traverse(f.geometry.coordinates);
    }
  }

  if (minX === Infinity) return null;
  return { minX, minY, maxX, maxY };
}
