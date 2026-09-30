/**
 * frontend/src/components/MapView.tsx
 *
 * Primary MapLibre GL JS interactive GIS component with client-side proj4 re-projection,
 * multi-layer rendering (AI Buildings, Physical Road Corridors, Preliminary Parcel Candidates),
 * Parcel Verification workflow (Phase 14B):
 *  - Dynamic status visualization (unreviewed, needs_review, accepted, edited, rejected)
 *  - Parcel candidate selection & inspection
 *  - Interactive polygon vertex editing (drag vertex, add midpoint, remove vertex)
 *  - Original AI vs human-edited geometry comparison overlay
 *  - Parcel attribute filtering (status, confidence band, uncertainty flags)
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import maplibregl from 'maplibre-gl';
import {
  GeoJSONFeatureCollection,
  AnalysisResponse,
  VisualizationMode,
  LayerViewMode,
  RoadSource,
  ParcelFilterMode,
  ReviewStatus,
} from '../types/analysis';
import {
  reprojectGeoJSONForDisplay,
  reprojectGeometryForDisplay,
  calculateBounds,
} from '../utils/geojson';
import { formatArea, formatConfidence } from '../utils/formatting';
import {
  Layers,
  Compass,
  CheckSquare,
  Square,
  Filter,
  Edit3,
  Save,
  X,
  RotateCcw,
  CheckCircle,
  AlertTriangle,
  Clock,
  XCircle,
  HelpCircle,
} from 'lucide-react';

interface MapViewProps {
  geojson: GeoJSONFeatureCollection | null;
  roadsGeojson?: GeoJSONFeatureCollection | null;
  parcelsGeojson?: GeoJSONFeatureCollection | null;
  analysis: AnalysisResponse | null;
  selectedBuildingId: number | null;
  onSelectBuilding: (buildingId: number) => void;
  selectedParcelId: number | null;
  onSelectParcel: (parcelId: number) => void;
  visualizationMode: VisualizationMode;
  onChangeVisualizationMode: (mode: VisualizationMode) => void;
  layerMode: LayerViewMode;
  onChangeLayerMode: (mode: LayerViewMode) => void;
  roadSource?: RoadSource;
  onChangeRoadSource?: (source: RoadSource) => void;
  parcelFilterMode: ParcelFilterMode;
  onChangeParcelFilterMode: (mode: ParcelFilterMode) => void;
  isEditingGeometry: boolean;
  onSaveEditedGeometry: (candidateId: number, newGeometry: any) => Promise<void>;
  onCancelEditGeometry: () => void;
  showOriginalGeometry: boolean;
  originalGeometry: any | null;
}

// Clean Dark Matter basemap style without requiring external API keys
const DARK_MATTER_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    'carto-dark': {
      type: 'raster',
      tiles: [
        'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
        'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
        'https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
      ],
      tileSize: 256,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>',
    },
  },
  layers: [
    {
      id: 'carto-dark-layer',
      type: 'raster',
      source: 'carto-dark',
      minzoom: 0,
      maxzoom: 20,
    },
  ],
};

export const MapView: React.FC<MapViewProps> = ({
  geojson,
  roadsGeojson,
  parcelsGeojson,
  analysis,
  selectedBuildingId,
  onSelectBuilding,
  selectedParcelId,
  onSelectParcel,
  visualizationMode,
  onChangeVisualizationMode,
  layerMode,
  onChangeLayerMode,
  roadSource = 'classical',
  onChangeRoadSource,
  parcelFilterMode,
  onChangeParcelFilterMode,
  isEditingGeometry,
  onSaveEditedGeometry,
  onCancelEditGeometry,
  showOriginalGeometry,
  originalGeometry,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const popupRef = useRef<maplibregl.Popup | null>(null);
  const boundsFittedRef = useRef<boolean>(false);
  const [mapLoaded, setMapLoaded] = useState(false);

  // Reprojected layers (WGS84 EPSG:4326 for web map display)
  const [reprojectedBuildings, setReprojectedBuildings] = useState<GeoJSONFeatureCollection | null>(null);
  const [reprojectedRoads, setReprojectedRoads] = useState<GeoJSONFeatureCollection | null>(null);
  const [reprojectedParcels, setReprojectedParcels] = useState<GeoJSONFeatureCollection | null>(null);

  // Layer visibility toggles
  const [showBuildings, setShowBuildings] = useState<boolean>(true);
  const [showRoads, setShowRoads] = useState<boolean>(true);
  const [showParcels, setShowParcels] = useState<boolean>(true);

  // Geometry editing local state
  const [editCoords, setEditCoords] = useState<[number, number][]>([]);
  const [activeDragIndex, setActiveDragIndex] = useState<number | null>(null);
  const [isSavingEdit, setIsSavingEdit] = useState<boolean>(false);

  // 1. Initialize MapLibre map instance
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: DARK_MATTER_STYLE,
      center: [12.0, 9.0], // Initial overview center
      zoom: 3,
      maxZoom: 22,
    });

    map.addControl(new maplibregl.NavigationControl({ showCompass: true }), 'top-right');
    map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-right');

    const popup = new maplibregl.Popup({
      closeButton: false,
      closeOnClick: false,
      offset: 12,
      maxWidth: '320px',
    });
    popupRef.current = popup;

    map.on('load', () => {
      setMapLoaded(true);
    });

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // 2. Client-side reprojection pipeline (EPSG:32732 -> EPSG:4326)
  useEffect(() => {
    const crsStr = analysis?.crs || 'EPSG:32732';

    if (geojson && analysis?.georeferenced) {
      setReprojectedBuildings(reprojectGeoJSONForDisplay(geojson, crsStr));
    } else {
      setReprojectedBuildings(geojson);
    }

    if (roadsGeojson && analysis?.georeferenced) {
      setReprojectedRoads(reprojectGeoJSONForDisplay(roadsGeojson, crsStr));
    } else {
      setReprojectedRoads(roadsGeojson || null);
    }

    if (parcelsGeojson && analysis?.georeferenced) {
      setReprojectedParcels(reprojectGeoJSONForDisplay(parcelsGeojson, crsStr));
    } else {
      setReprojectedParcels(parcelsGeojson || null);
    }
  }, [geojson, roadsGeojson, parcelsGeojson, analysis]);

  // 3. Update or create MapLibre GeoJSON sources and layers
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;

    // Helper: Add or update source
    const syncSource = (id: string, data: any) => {
      const src = map.getSource(id) as maplibregl.GeoJSONSource | undefined;
      if (src) {
        src.setData(data || { type: 'FeatureCollection', features: [] });
      } else if (data) {
        map.addSource(id, {
          type: 'geojson',
          data: data,
        });
      }
    };

    // Sync primary vector sources
    syncSource('parcels-source', reprojectedParcels);
    syncSource('roads-source', reprojectedRoads);
    syncSource('buildings-source', reprojectedBuildings);

    // Layer 1: Preliminary Parcels (Bottom)
    if (map.getSource('parcels-source') && !map.getLayer('parcels-fill')) {
      // 1.1 Parcels Fill with Status-Driven Coloring (Step 7)
      map.addLayer({
        id: 'parcels-fill',
        type: 'fill',
        source: 'parcels-source',
        paint: {
          'fill-color': [
            'case',
            ['==', ['get', 'review_status'], 'accepted'],
            '#10B981', // emerald
            ['==', ['get', 'review_status'], 'edited'],
            '#3B82F6', // blue
            ['==', ['get', 'review_status'], 'needs_review'],
            '#F59E0B', // amber
            ['==', ['get', 'review_status'], 'rejected'],
            '#EF4444', // red
            '#00FFA3', // unreviewed (cyan-green default)
          ],
          'fill-opacity': [
            'case',
            ['==', ['get', 'review_status'], 'rejected'],
            0.12,
            ['==', ['get', 'review_status'], 'edited'],
            0.35,
            ['==', ['get', 'review_status'], 'accepted'],
            0.32,
            ['==', ['get', 'review_status'], 'needs_review'],
            0.32,
            0.22,
          ],
        },
      });

      // 1.2 Parcels Boundary Outline
      map.addLayer({
        id: 'parcels-outline',
        type: 'line',
        source: 'parcels-source',
        paint: {
          'line-color': [
            'case',
            ['==', ['get', 'review_status'], 'accepted'],
            '#059669',
            ['==', ['get', 'review_status'], 'edited'],
            '#2563EB',
            ['==', ['get', 'review_status'], 'needs_review'],
            '#D97706',
            ['==', ['get', 'review_status'], 'rejected'],
            '#DC2626',
            '#00E676',
          ],
          'line-width': [
            'case',
            ['==', ['get', 'review_status'], 'edited'],
            2.0,
            ['==', ['get', 'review_status'], 'accepted'],
            1.8,
            1.4,
          ],
          'line-dasharray': [
            'case',
            ['==', ['get', 'review_status'], 'accepted'],
            ['literal', [1, 0]], // solid
            ['==', ['get', 'review_status'], 'edited'],
            ['literal', [1, 0]], // solid
            ['literal', [3, 2]], // dashed for unreviewed/needs_review
          ],
        },
      });

      // 1.3 Selected Parcel Highlighting Ring
      map.addLayer({
        id: 'parcels-selected-outline',
        type: 'line',
        source: 'parcels-source',
        filter: ['==', ['coalesce', ['get', 'candidate_id'], ['get', 'parcel_id'], -1], -1],
        paint: {
          'line-color': '#FACC15',
          'line-width': 3.5,
        },
      });

      // Hover on parcel (Rich Intelligence Tooltip)
      map.on('mousemove', 'parcels-fill', (e) => {
        if (!e.features || e.features.length === 0) return;
        map.getCanvas().style.cursor = 'pointer';
        const p = e.features[0].properties as any;
        const candidateId = p.candidate_id || p.parcel_id || 'N/A';
        const status = (p.review_status || 'unreviewed').toUpperCase().replace(/_/g, ' ');
        const score = p.confidence !== undefined
          ? `${(parseFloat(p.confidence) * 100).toFixed(1)}%`
          : p.candidate_score
          ? `${(parseFloat(p.candidate_score) * 100).toFixed(1)}%`
          : 'N/A';
        const band = p.confidence_band || 'HIGH';
        const frontage = p.road_frontage !== undefined
          ? `${parseFloat(p.road_frontage).toFixed(1)} m`
          : p.road_frontage_m
          ? `${parseFloat(p.road_frontage_m).toFixed(1)} m`
          : '0.0 m';
        const dist = p.nearest_road_distance !== undefined
          ? `${parseFloat(p.nearest_road_distance).toFixed(1)} m`
          : p.road_distance_m !== undefined && p.road_distance_m !== null
          ? `${parseFloat(p.road_distance_m).toFixed(1)} m`
          : '0 m';
        const cov = p.building_coverage !== undefined
          ? `${(parseFloat(p.building_coverage) * 100).toFixed(1)}%`
          : p.coverage_ratio
          ? `${(parseFloat(p.coverage_ratio) * 100).toFixed(1)}%`
          : 'N/A';
        const flags = p.flags ? (Array.isArray(p.flags) ? p.flags.join(', ') : p.flags) : '';

        const statusColor =
          status === 'ACCEPTED'
            ? 'text-emerald-400'
            : status === 'EDITED'
            ? 'text-blue-400'
            : status === 'NEEDS REVIEW'
            ? 'text-amber-400'
            : status === 'REJECTED'
            ? 'text-red-400'
            : 'text-gray-300';

        popupRef.current
          ?.setLngLat(e.lngLat)
          .setHTML(
            `<div class="text-xs font-mono space-y-1 p-1">
              <div class="font-bold flex items-center justify-between border-b border-gray-700 pb-1">
                <span class="text-white">Parcel P${String(candidateId).padStart(3, '0')}</span>
                <span class="text-[10px] font-bold ${statusColor}">[${status}]</span>
              </div>
              <div class="text-gis-textPrimary flex justify-between">
                <span>Confidence:</span>
                <strong class="text-gis-cyan">${score} (${band})</strong>
              </div>
              <div class="text-gis-textPrimary flex justify-between">
                <span>Building Coverage:</span>
                <strong>${cov}</strong>
              </div>
              <div class="text-gis-textPrimary flex justify-between">
                <span>Road Frontage:</span>
                <strong class="text-cyan-300">${frontage}</strong>
              </div>
              <div class="text-gis-textPrimary flex justify-between">
                <span>Nearest Road:</span>
                <strong>${dist}</strong>
              </div>
              ${
                flags
                  ? `<div class="text-[10px] text-yellow-300 pt-1 border-t border-gray-800">
                      ⚠ Flags: ${flags}
                    </div>`
                  : ''
              }
              <div class="text-[9px] text-gray-400 italic pt-1 border-t border-gray-800 leading-tight">
                Preliminary AI candidate; requires survey verification. Not legal boundary.
              </div>
            </div>`
          )
          .addTo(map);
      });

      map.on('mouseleave', 'parcels-fill', () => {
        map.getCanvas().style.cursor = '';
        popupRef.current?.remove();
      });

      // Click on parcel candidate (Selection Handler)
      map.on('click', 'parcels-fill', (e) => {
        if (!e.features || e.features.length === 0) return;
        const props = e.features[0].properties as any;
        const cid = props.candidate_id || props.parcel_id;
        if (cid !== undefined) {
          onSelectParcel(Number(cid));
        }
      });
    }

    // Layer 2: Physical Road Corridors (Middle)
    if (map.getSource('roads-source') && !map.getLayer('roads-fill')) {
      const roadColor =
        roadSource === 'classical' ? '#00D2FF' : roadSource === 'deep_learning' ? '#E040FB' : '#00E676';
      const roadOutline =
        roadSource === 'classical' ? '#0066FF' : roadSource === 'deep_learning' ? '#AA00FF' : '#00B0FF';

      map.addLayer({
        id: 'roads-fill',
        type: 'fill',
        source: 'roads-source',
        paint: {
          'fill-color': roadColor,
          'fill-opacity': 0.45,
        },
      });

      map.addLayer({
        id: 'roads-outline',
        type: 'line',
        source: 'roads-source',
        paint: {
          'line-color': roadOutline,
          'line-width': 1.8,
        },
      });

      // Hover on road
      map.on('mousemove', 'roads-fill', (e) => {
        if (!e.features || e.features.length === 0) return;
        map.getCanvas().style.cursor = 'pointer';
        const r = e.features[0].properties as any;
        const width = r.estimated_width_m || r.width_estimate_m ? `${parseFloat(r.estimated_width_m || r.width_estimate_m).toFixed(1)} m` : '3.7 m';
        const len = r.length_m ? `${parseFloat(r.length_m).toFixed(1)} m` : 'N/A';
        const sourceEvidence = r.source_evidence || r.source || (roadSource === 'hybrid' ? 'Hybrid Fusion' : roadSource);

        popupRef.current
          ?.setLngLat(e.lngLat)
          .setHTML(
            `<div class="text-xs font-mono space-y-1 p-1">
              <div class="font-bold text-cyan-400 flex items-center justify-between border-b border-gray-700 pb-1">
                <span>Road Corridor #${r.road_id || '1'}</span>
              </div>
              <div class="text-gis-textPrimary">Length: <strong>${len}</strong></div>
              <div class="text-gis-textPrimary">Est. Width: <strong>${width}</strong></div>
              <div class="text-gis-textPrimary">Evidence: <strong class="text-emerald-400 uppercase text-[10px]">${sourceEvidence}</strong></div>
              <div class="text-[10px] text-gray-400 italic pt-1 border-t border-gray-800">Physical access corridor; not legal right-of-way.</div>
            </div>`
          )
          .addTo(map);
      });

      map.on('mouseleave', 'roads-fill', () => {
        map.getCanvas().style.cursor = '';
        popupRef.current?.remove();
      });
    }

    // Layer 3: AI Buildings (Top)
    if (map.getSource('buildings-source') && !map.getLayer('buildings-fill')) {
      map.addLayer({
        id: 'buildings-fill',
        type: 'fill',
        source: 'buildings-source',
        paint: {
          'fill-color': [
            'case',
            ['==', ['get', 'id'], selectedBuildingId ?? -1],
            '#FACC15',
            '#FF9100',
          ],
          'fill-opacity': 0.65,
        },
      });

      map.addLayer({
        id: 'buildings-outline',
        type: 'line',
        source: 'buildings-source',
        paint: {
          'line-color': '#D50000',
          'line-width': 1.4,
        },
      });

      map.addLayer({
        id: 'buildings-selected-outline',
        type: 'line',
        source: 'buildings-source',
        filter: ['==', ['get', 'id'], -1],
        paint: {
          'line-color': '#FACC15',
          'line-width': 3,
        },
      });

      // Hover on building
      map.on('mousemove', 'buildings-fill', (e) => {
        if (!e.features || e.features.length === 0) return;
        map.getCanvas().style.cursor = 'pointer';

        const props = e.features[0].properties as any;
        const areaVal = props.area ? parseFloat(props.area) : 0;
        const confVal = props.confidence ? parseFloat(props.confidence) : 0;
        const unit = props.area_unit || 'm²';

        popupRef.current
          ?.setLngLat(e.lngLat)
          .setHTML(
            `<div class="text-xs font-mono space-y-1 p-1">
              <div class="font-bold text-yellow-400 flex items-center justify-between border-b border-gray-700 pb-1">
                <span>Building Footprint #${props.id}</span>
              </div>
              <div class="text-gis-textPrimary">Area: <strong>${formatArea(areaVal, unit)}</strong></div>
              <div class="text-gis-textMuted">Confidence: <span class="text-gis-success font-semibold">${formatConfidence(confVal)}</span></div>
            </div>`
          )
          .addTo(map);
      });

      map.on('mouseleave', 'buildings-fill', () => {
        map.getCanvas().style.cursor = '';
        popupRef.current?.remove();
      });

      // Click selection handler for buildings
      map.on('click', 'buildings-fill', (e) => {
        if (!e.features || e.features.length === 0) return;
        const id = e.features[0].properties?.id;
        if (id !== undefined) {
          onSelectBuilding(Number(id));
        }
      });
    }

    // Auto-fit bounds on initial load
    const targetData = reprojectedBuildings || reprojectedParcels || reprojectedRoads;
    if (targetData && !boundsFittedRef.current) {
      const bounds = calculateBounds(targetData);
      if (bounds) {
        map.fitBounds(
          [
            [bounds.minX, bounds.minY],
            [bounds.maxX, bounds.maxY],
          ],
          { padding: 60, maxZoom: 19, duration: 1000 }
        );
        boundsFittedRef.current = true;
      }
    }
  }, [reprojectedBuildings, reprojectedRoads, reprojectedParcels, mapLoaded]);

  // 4. Update Selected Parcel Outline Ring
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded || !map.getLayer('parcels-selected-outline')) return;

    map.setFilter('parcels-selected-outline', [
      '==',
      ['coalesce', ['get', 'candidate_id'], ['get', 'parcel_id'], -1],
      selectedParcelId ?? -1,
    ]);
  }, [selectedParcelId, mapLoaded]);

  // 5. Update Selected Building Outline Ring
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded || !map.getLayer('buildings-selected-outline')) return;

    map.setFilter('buildings-selected-outline', ['==', ['get', 'id'], selectedBuildingId ?? -1]);
  }, [selectedBuildingId, mapLoaded]);

  // 6. Handle Parcel Filter Application (Step 8)
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded || !map.getLayer('parcels-fill')) return;

    let filterExpression: any = null;

    switch (parcelFilterMode) {
      case 'unreviewed':
        filterExpression = ['==', ['coalesce', ['get', 'review_status'], 'unreviewed'], 'unreviewed'];
        break;
      case 'needs_review':
        filterExpression = ['==', ['get', 'review_status'], 'needs_review'];
        break;
      case 'accepted':
        filterExpression = ['==', ['get', 'review_status'], 'accepted'];
        break;
      case 'edited':
        filterExpression = ['==', ['get', 'review_status'], 'edited'];
        break;
      case 'rejected':
        filterExpression = ['==', ['get', 'review_status'], 'rejected'];
        break;
      case 'high_confidence':
        filterExpression = ['>=', ['coalesce', ['get', 'confidence'], ['get', 'candidate_score'], 0], 0.75];
        break;
      case 'moderate_confidence':
        filterExpression = ['<', ['coalesce', ['get', 'confidence'], ['get', 'candidate_score'], 0], 0.75];
        break;
      case 'has_flags':
        filterExpression = ['!=', ['coalesce', ['get', 'flags'], ''], ''];
        break;
      case 'all':
      default:
        filterExpression = null;
        break;
    }

    map.setFilter('parcels-fill', filterExpression);
    map.setFilter('parcels-outline', filterExpression);
  }, [parcelFilterMode, mapLoaded]);

  // 7. Layer Visibility Toggles
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;

    if (map.getLayer('buildings-fill')) {
      map.setLayoutProperty('buildings-fill', 'visibility', showBuildings ? 'visible' : 'none');
      map.setLayoutProperty('buildings-outline', 'visibility', showBuildings ? 'visible' : 'none');
      if (map.getLayer('buildings-selected-outline')) {
        map.setLayoutProperty('buildings-selected-outline', 'visibility', showBuildings ? 'visible' : 'none');
      }
    }

    if (map.getLayer('roads-fill')) {
      map.setLayoutProperty('roads-fill', 'visibility', showRoads ? 'visible' : 'none');
      map.setLayoutProperty('roads-outline', 'visibility', showRoads ? 'visible' : 'none');
    }

    if (map.getLayer('parcels-fill')) {
      map.setLayoutProperty('parcels-fill', 'visibility', showParcels ? 'visible' : 'none');
      map.setLayoutProperty('parcels-outline', 'visibility', showParcels ? 'visible' : 'none');
      if (map.getLayer('parcels-selected-outline')) {
        map.setLayoutProperty('parcels-selected-outline', 'visibility', showParcels ? 'visible' : 'none');
      }
    }
  }, [showBuildings, showRoads, showParcels, mapLoaded]);

  // 8. Building Confidence Heat-Map Styling
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded || !map.getLayer('buildings-fill')) return;

    if (visualizationMode === 'confidence') {
      map.setPaintProperty('buildings-fill', 'fill-color', [
        'case',
        ['==', ['get', 'id'], selectedBuildingId ?? -1],
        '#FACC15',
        ['>=', ['get', 'confidence'], 0.8],
        '#10B981', // High: emerald
        ['>=', ['get', 'confidence'], 0.6],
        '#F59E0B', // Medium: amber
        '#EF4444', // Low: red
      ]);
    } else {
      map.setPaintProperty('buildings-fill', 'fill-color', [
        'case',
        ['==', ['get', 'id'], selectedBuildingId ?? -1],
        '#FACC15',
        '#FF9100',
      ]);
    }
  }, [visualizationMode, selectedBuildingId, mapLoaded]);

  // ════════════════════════════════════════════════════════════════════
  // 9. Ghost Original Geometry Overlay (Step 13: Original vs Edited)
  // ════════════════════════════════════════════════════════════════════
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;

    const ghostSourceId = 'ghost-original-source';

    if (showOriginalGeometry && originalGeometry) {
      const crsStr = analysis?.crs || 'EPSG:32732';
      const reprojectedGhost = reprojectGeometryForDisplay(originalGeometry, crsStr);

      const ghostData = {
        type: 'FeatureCollection',
        features: [
          {
            type: 'Feature',
            properties: { label: 'Original AI Hypothesis' },
            geometry: reprojectedGhost,
          },
        ],
      };

      const src = map.getSource(ghostSourceId) as maplibregl.GeoJSONSource | undefined;
      if (src) {
        src.setData(ghostData as any);
      } else {
        map.addSource(ghostSourceId, {
          type: 'geojson',
          data: ghostData as any,
        });

        map.addLayer({
          id: 'ghost-original-fill',
          type: 'fill',
          source: ghostSourceId,
          paint: {
            'fill-color': '#F59E0B',
            'fill-opacity': 0.25,
          },
        });

        map.addLayer({
          id: 'ghost-original-line',
          type: 'line',
          source: ghostSourceId,
          paint: {
            'line-color': '#F59E0B',
            'line-width': 2.5,
            'line-dasharray': [4, 2],
          },
        });
      }
    } else {
      if (map.getLayer('ghost-original-line')) map.removeLayer('ghost-original-line');
      if (map.getLayer('ghost-original-fill')) map.removeLayer('ghost-original-fill');
      if (map.getSource(ghostSourceId)) map.removeSource(ghostSourceId);
    }
  }, [showOriginalGeometry, originalGeometry, analysis, mapLoaded]);

  // ════════════════════════════════════════════════════════════════════
  // 10. Lightweight Geometry Editing Pipeline (Step 5)
  // ════════════════════════════════════════════════════════════════════
  // Initialize editable coordinates when entering edit mode
  useEffect(() => {
    if (!isEditingGeometry || !selectedParcelId || !reprojectedParcels) {
      setEditCoords([]);
      setActiveDragIndex(null);
      return;
    }

    const feature = reprojectedParcels.features.find((f: any) => {
      const cid = f.properties?.candidate_id || f.properties?.parcel_id;
      return Number(cid) === selectedParcelId;
    });

    if (feature && feature.geometry) {
      let rawCoords: any = feature.geometry.coordinates;
      if (feature.geometry.type === 'MultiPolygon' && rawCoords[0]) {
        rawCoords = rawCoords[0][0];
      } else if (feature.geometry.type === 'Polygon' && rawCoords[0]) {
        rawCoords = rawCoords[0];
      }

      if (Array.isArray(rawCoords) && rawCoords.length > 0) {
        setEditCoords(rawCoords.map((pt: any) => [pt[0], pt[1]]));
      }
    }
  }, [isEditingGeometry, selectedParcelId, reprojectedParcels]);

  // Sync editing layers (live polygon + vertex handles + midpoint handles)
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;

    const editPolySourceId = 'editing-parcel-source';
    const editVertexSourceId = 'editing-vertices-source';

    if (isEditingGeometry && editCoords.length >= 3) {
      // 1. Polygon Feature
      const polyData = {
        type: 'FeatureCollection',
        features: [
          {
            type: 'Feature',
            properties: {},
            geometry: {
              type: 'Polygon',
              coordinates: [editCoords],
            },
          },
        ],
      };

      // 2. Vertex Points & Midpoint Points
      const vertexFeatures: any[] = [];

      // Add main vertices (exclude duplicate closing vertex for handles)
      const count = editCoords.length > 1 && editCoords[0][0] === editCoords[editCoords.length - 1][0]
        ? editCoords.length - 1
        : editCoords.length;

      for (let i = 0; i < count; i++) {
        vertexFeatures.push({
          type: 'Feature',
          properties: { index: i, type: 'vertex' },
          geometry: {
            type: 'Point',
            coordinates: editCoords[i],
          },
        });

        // Add midpoint between vertex i and i+1
        const nextIdx = (i + 1) % count;
        const midLon = (editCoords[i][0] + editCoords[nextIdx][0]) / 2;
        const midLat = (editCoords[i][1] + editCoords[nextIdx][1]) / 2;

        vertexFeatures.push({
          type: 'Feature',
          properties: { index: i, type: 'midpoint' },
          geometry: {
            type: 'Point',
            coordinates: [midLon, midLat],
          },
        });
      }

      const vertexData = {
        type: 'FeatureCollection',
        features: vertexFeatures,
      };

      // Sync polygon source
      const polySrc = map.getSource(editPolySourceId) as maplibregl.GeoJSONSource | undefined;
      if (polySrc) {
        polySrc.setData(polyData as any);
      } else {
        map.addSource(editPolySourceId, { type: 'geojson', data: polyData as any });
        map.addLayer({
          id: 'editing-parcel-fill',
          type: 'fill',
          source: editPolySourceId,
          paint: {
            'fill-color': '#3B82F6',
            'fill-opacity': 0.45,
          },
        });
        map.addLayer({
          id: 'editing-parcel-line',
          type: 'line',
          source: editPolySourceId,
          paint: {
            'line-color': '#60A5FA',
            'line-width': 2.5,
          },
        });
      }

      // Sync vertex source
      const vertSrc = map.getSource(editVertexSourceId) as maplibregl.GeoJSONSource | undefined;
      if (vertSrc) {
        vertSrc.setData(vertexData as any);
      } else {
        map.addSource(editVertexSourceId, { type: 'geojson', data: vertexData as any });

        // Midpoints layer (cyan small dots)
        map.addLayer({
          id: 'editing-midpoints-layer',
          type: 'circle',
          source: editVertexSourceId,
          filter: ['==', ['get', 'type'], 'midpoint'],
          paint: {
            'circle-radius': 4.5,
            'circle-color': '#38BDF8',
            'circle-stroke-color': '#0369A1',
            'circle-stroke-width': 1.5,
            'circle-opacity': 0.85,
          },
        });

        // Vertices layer (white circle with blue stroke)
        map.addLayer({
          id: 'editing-vertices-layer',
          type: 'circle',
          source: editVertexSourceId,
          filter: ['==', ['get', 'type'], 'vertex'],
          paint: {
            'circle-radius': 6.5,
            'circle-color': '#FFFFFF',
            'circle-stroke-color': '#2563EB',
            'circle-stroke-width': 2.5,
          },
        });
      }
    } else {
      // Clean up editing layers when edit mode ends
      if (map.getLayer('editing-midpoints-layer')) map.removeLayer('editing-midpoints-layer');
      if (map.getLayer('editing-vertices-layer')) map.removeLayer('editing-vertices-layer');
      if (map.getSource(editVertexSourceId)) map.removeSource(editVertexSourceId);

      if (map.getLayer('editing-parcel-line')) map.removeLayer('editing-parcel-line');
      if (map.getLayer('editing-parcel-fill')) map.removeLayer('editing-parcel-fill');
      if (map.getSource(editPolySourceId)) map.removeSource(editPolySourceId);
    }
  }, [isEditingGeometry, editCoords, mapLoaded]);

  // Vertex Drag and Click Interactivity
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded || !isEditingGeometry) return;

    const canvas = map.getCanvas();

    const onVertexMouseDown = (e: maplibregl.MapMouseEvent & { features?: maplibregl.MapGeoJSONFeature[] }) => {
      if (!e.features || e.features.length === 0) return;
      const feat = e.features[0];
      const props = feat.properties as any;

      if (props.type === 'midpoint') {
        // Clicking midpoint inserts a vertex!
        const insertIdx = props.index + 1;
        const newCoords = [...editCoords];
        const [lng, lat] = (feat.geometry as any).coordinates;
        newCoords.splice(insertIdx, 0, [lng, lat]);
        // Ensure closing vertex matches
        if (newCoords[0][0] !== newCoords[newCoords.length - 1][0] || newCoords[0][1] !== newCoords[newCoords.length - 1][1]) {
          newCoords[newCoords.length - 1] = [newCoords[0][0], newCoords[0][1]];
        }
        setEditCoords(newCoords);
        return;
      }

      if (props.type === 'vertex') {
        if (e.originalEvent.altKey || e.originalEvent.shiftKey) {
          // Alt+Click deletes a vertex if more than 3 distinct remain
          if (editCoords.length > 4) {
            const delIdx = props.index;
            const newCoords = [...editCoords];
            newCoords.splice(delIdx, 1);
            if (delIdx === 0) {
              newCoords[newCoords.length - 1] = [newCoords[0][0], newCoords[0][1]];
            }
            setEditCoords(newCoords);
          }
          return;
        }

        // Start dragging vertex
        map.dragPan.disable();
        setActiveDragIndex(props.index);
        canvas.style.cursor = 'grabbing';
      }
    };

    const onMouseMove = (e: maplibregl.MapMouseEvent) => {
      if (activeDragIndex === null) return;
      const lngLat = e.lngLat;
      setEditCoords((prevCoords) => {
        if (activeDragIndex >= prevCoords.length) return prevCoords;
        const next = [...prevCoords];
        next[activeDragIndex] = [lngLat.lng, lngLat.lat];
        // If first vertex moved, update closing vertex too
        if (activeDragIndex === 0) {
          next[next.length - 1] = [lngLat.lng, lngLat.lat];
        }
        return next;
      });
    };

    const onMouseUp = () => {
      if (activeDragIndex !== null) {
        setActiveDragIndex(null);
        map.dragPan.enable();
        canvas.style.cursor = '';
      }
    };

    map.on('mousedown', 'editing-vertices-layer', onVertexMouseDown);
    map.on('mousedown', 'editing-midpoints-layer', onVertexMouseDown);
    map.on('mousemove', onMouseMove);
    map.on('mouseup', onMouseUp);

    return () => {
      map.off('mousedown', 'editing-vertices-layer', onVertexMouseDown);
      map.off('mousedown', 'editing-midpoints-layer', onVertexMouseDown);
      map.off('mousemove', onMouseMove);
      map.off('mouseup', onMouseUp);
      map.dragPan.enable();
    };
  }, [isEditingGeometry, activeDragIndex, editCoords, mapLoaded]);

  // Handle Save Edited Geometry
  const handleSaveGeometryClick = async () => {
    if (!selectedParcelId || editCoords.length < 4) return;
    setIsSavingEdit(true);

    try {
      // Ensure closing vertex strictly matches start
      const finalCoords = [...editCoords];
      if (
        finalCoords[0][0] !== finalCoords[finalCoords.length - 1][0] ||
        finalCoords[0][1] !== finalCoords[finalCoords.length - 1][1]
      ) {
        finalCoords[finalCoords.length - 1] = [finalCoords[0][0], finalCoords[0][1]];
      }

      const newGeometry = {
        type: 'Polygon',
        coordinates: [finalCoords],
      };

      await onSaveEditedGeometry(selectedParcelId, newGeometry);
    } finally {
      setIsSavingEdit(false);
    }
  };

  // Recenter map handler
  const handleRecenter = () => {
    const map = mapRef.current;
    if (!map) return;
    const targetData = reprojectedParcels || reprojectedBuildings || reprojectedRoads;
    if (targetData) {
      const bounds = calculateBounds(targetData);
      if (bounds) {
        map.fitBounds(
          [
            [bounds.minX, bounds.minY],
            [bounds.maxX, bounds.maxY],
          ],
          { padding: 60, maxZoom: 19 }
        );
      }
    }
  };

  return (
    <div className="relative w-full h-full flex-1 overflow-hidden bg-gis-bg">
      {/* MapLibre DOM element */}
      <div ref={mapContainerRef} className="w-full h-full" />

      {/* Floating Active Geometry Edit Toolbar (Step 5) */}
      {isEditingGeometry && selectedParcelId && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-30 bg-gis-card/95 border-2 border-gis-accent backdrop-blur-md rounded-xl p-3 shadow-2xl flex items-center space-x-3 text-xs animate-in fade-in slide-in-from-top-3">
          <div className="flex items-center space-x-2">
            <div className="w-6 h-6 rounded bg-gis-accent/20 text-gis-accent flex items-center justify-center font-mono font-bold">
              <Edit3 className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="font-bold text-white uppercase text-[11px] tracking-wider flex items-center space-x-1.5">
                <span>Editing Parcel P{String(selectedParcelId).padStart(3, '0')}</span>
                <span className="text-[10px] text-gis-accent font-mono">
                  ({editCoords.length > 0 ? editCoords.length - 1 : 0} vertices)
                </span>
              </div>
              <div className="text-[10px] text-gis-textMuted flex items-center space-x-1">
                <HelpCircle className="w-3 h-3 text-gis-cyan" />
                <span>Drag vertex to move • Click midpoint to add • Alt+click vertex to delete</span>
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-1.5 pl-2 border-l border-gis-border">
            <button
              type="button"
              disabled={isSavingEdit}
              onClick={handleSaveGeometryClick}
              className="py-1.5 px-3 bg-gis-accent hover:bg-gis-accent/80 text-white font-bold rounded text-xs transition-colors flex items-center space-x-1 shadow-md"
            >
              <Save className="w-3.5 h-3.5" />
              <span>{isSavingEdit ? 'Saving...' : 'Save Geometry'}</span>
            </button>
            <button
              type="button"
              onClick={onCancelEditGeometry}
              className="py-1.5 px-2.5 bg-gis-surface hover:bg-gis-surface/80 border border-gis-border text-gis-textMuted hover:text-white rounded text-xs font-semibold transition-colors flex items-center space-x-1"
            >
              <X className="w-3.5 h-3.5" />
              <span>Cancel</span>
            </button>
          </div>
        </div>
      )}

      {/* Floating Map Controls & Legends */}
      <div className="absolute top-4 left-4 z-10 flex flex-col space-y-2">
        {/* Layer Visibility & Filter Control Box */}
        <div className="bg-gis-card/95 border border-gis-border backdrop-blur-md rounded-lg p-3 shadow-xl text-xs space-y-2.5 max-w-xs">
          <div className="flex items-center justify-between border-b border-gis-border pb-1.5">
            <span className="font-bold text-white uppercase text-[11px] tracking-wider flex items-center space-x-1.5">
              <Layers className="w-3.5 h-3.5 text-gis-cyan" />
              <span>GIS Layers</span>
            </span>
            <span className="text-[10px] text-gis-cyan font-mono font-semibold">Phase 14B</span>
          </div>

          {/* Road Source Selector Toggle (Phase 13D 3-way: Classical, Deep Learning, Hybrid) */}
          <div className="bg-gis-surface/60 rounded p-2 border border-gis-border/70 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-semibold text-gis-textMuted uppercase tracking-wider">
                Road Model Source
              </span>
              <span className="text-[9px] px-1.5 py-0.2 rounded bg-gis-surface text-emerald-400 font-mono">
                {roadSource === 'classical'
                  ? 'CV Baseline'
                  : roadSource === 'deep_learning'
                  ? 'DL U-Net'
                  : 'Hybrid Fusion'}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-1 bg-gis-card p-0.5 rounded border border-gis-border">
              <button
                type="button"
                onClick={() => onChangeRoadSource?.('classical')}
                className={`px-1.5 py-1 rounded text-[10px] font-medium transition-all ${
                  roadSource === 'classical'
                    ? 'bg-gis-cyan text-gis-bg font-bold shadow-sm'
                    : 'text-gis-textMuted hover:text-white'
                }`}
              >
                Classical
              </button>
              <button
                type="button"
                onClick={() => onChangeRoadSource?.('deep_learning')}
                className={`px-1.5 py-1 rounded text-[10px] font-medium transition-all ${
                  roadSource === 'deep_learning'
                    ? 'bg-gis-accent text-white font-bold shadow-sm'
                    : 'text-gis-textMuted hover:text-white'
                }`}
              >
                Neural
              </button>
              <button
                type="button"
                onClick={() => onChangeRoadSource?.('hybrid')}
                className={`px-1.5 py-1 rounded text-[10px] font-medium transition-all ${
                  roadSource === 'hybrid'
                    ? 'bg-emerald-500 text-gis-bg font-bold shadow-sm'
                    : 'text-gis-textMuted hover:text-white'
                }`}
              >
                Hybrid
              </button>
            </div>
          </div>

          {/* Layer Toggles */}
          <div className="space-y-1 font-medium text-[11px]">
            {/* Buildings Toggle */}
            <button
              onClick={() => setShowBuildings(!showBuildings)}
              className="w-full flex items-center justify-between p-1.5 rounded hover:bg-gis-surface/80 text-left transition-colors"
            >
              <div className="flex items-center space-x-2">
                <span className="w-3 h-3 rounded-sm bg-[#FF9100] border border-[#D50000]"></span>
                <span className={showBuildings ? 'text-white' : 'text-gis-textMuted line-through'}>
                  AI Building Footprints
                </span>
              </div>
              {showBuildings ? (
                <CheckSquare className="w-3.5 h-3.5 text-gis-cyan" />
              ) : (
                <Square className="w-3.5 h-3.5 text-gis-textMuted" />
              )}
            </button>

            {/* Roads Toggle */}
            <button
              onClick={() => setShowRoads(!showRoads)}
              className="w-full flex items-center justify-between p-1.5 rounded hover:bg-gis-surface/80 text-left transition-colors"
            >
              <div className="flex items-center space-x-2">
                <span
                  className={`w-3 h-3 rounded-sm ${
                    roadSource === 'classical'
                      ? 'bg-[#00D2FF] border border-[#0066FF]'
                      : roadSource === 'deep_learning'
                      ? 'bg-[#E040FB] border border-[#AA00FF]'
                      : 'bg-[#00E676] border border-[#00B0FF]'
                  }`}
                ></span>
                <span className={showRoads ? 'text-white' : 'text-gis-textMuted line-through'}>
                  {roadSource === 'classical'
                    ? 'Road Corridors (CV)'
                    : roadSource === 'deep_learning'
                    ? 'Neural Roads (DL)'
                    : 'Hybrid Roads (Fusion)'}
                </span>
              </div>
              {showRoads ? (
                <CheckSquare className="w-3.5 h-3.5 text-gis-cyan" />
              ) : (
                <Square className="w-3.5 h-3.5 text-gis-textMuted" />
              )}
            </button>

            {/* Preliminary Parcels Toggle */}
            <button
              onClick={() => setShowParcels(!showParcels)}
              className="w-full flex items-center justify-between p-1.5 rounded hover:bg-gis-surface/80 text-left transition-colors"
            >
              <div className="flex items-center space-x-2">
                <span className="w-3 h-3 rounded-sm bg-[#00FFA3] border border-[#00E676]"></span>
                <span className={showParcels ? 'text-white' : 'text-gis-textMuted line-through'}>
                  Preliminary Parcel Candidates
                </span>
              </div>
              {showParcels ? (
                <CheckSquare className="w-3.5 h-3.5 text-gis-cyan" />
              ) : (
                <Square className="w-3.5 h-3.5 text-gis-textMuted" />
              )}
            </button>
          </div>

          {/* Parcel Review Filter Selector (Step 8) */}
          <div className="pt-2 border-t border-gis-border/70 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold text-white uppercase tracking-wider flex items-center space-x-1">
                <Filter className="w-3 h-3 text-gis-cyan" />
                <span>Parcel Filter</span>
              </span>
              <span className="text-[9px] font-mono text-gis-cyan">
                {parcelFilterMode.replace(/_/g, ' ').toUpperCase()}
              </span>
            </div>

            <select
              value={parcelFilterMode}
              onChange={(e) => onChangeParcelFilterMode(e.target.value as ParcelFilterMode)}
              className="w-full bg-gis-surface border border-gis-border rounded p-1.5 text-xs text-white focus:outline-none focus:border-gis-cyan font-sans"
            >
              <option value="all">All Candidates (254)</option>
              <option value="unreviewed">Unreviewed</option>
              <option value="needs_review">Needs Review</option>
              <option value="accepted">Accepted</option>
              <option value="edited">Edited</option>
              <option value="rejected">Rejected</option>
              <option value="high_confidence">High Confidence (≥75%)</option>
              <option value="moderate_confidence">Moderate Confidence (&lt;75%)</option>
              <option value="has_flags">Has Uncertainty Flags</option>
            </select>
          </div>

          {/* Review Status Legend (Step 7) */}
          <div className="pt-2 border-t border-gis-border/70 space-y-1">
            <span className="text-[9px] uppercase font-bold text-gis-textMuted tracking-wider block">
              Review Status Legend
            </span>
            <div className="grid grid-cols-2 gap-1 text-[10px] font-mono">
              <div className="flex items-center space-x-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-[#00FFA3] border border-[#00E676]" />
                <span className="text-gray-300">Unreviewed</span>
              </div>
              <div className="flex items-center space-x-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-[#F59E0B] border border-[#D97706]" />
                <span className="text-amber-400">Needs Review</span>
              </div>
              <div className="flex items-center space-x-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-[#10B981] border border-[#059669]" />
                <span className="text-emerald-400">Accepted</span>
              </div>
              <div className="flex items-center space-x-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-[#3B82F6] border border-[#2563EB]" />
                <span className="text-blue-400">Edited</span>
              </div>
              <div className="flex items-center space-x-1.5 col-span-2">
                <span className="w-2.5 h-2.5 rounded-sm bg-[#EF4444] border border-[#DC2626]" />
                <span className="text-red-400">Rejected</span>
              </div>
            </div>
          </div>
        </div>

        {/* Building Color Mode Toggle */}
        <div className="bg-gis-card/90 border border-gis-border backdrop-blur-md rounded-lg p-2.5 shadow-lg text-xs space-y-2">
          <div className="flex items-center justify-between space-x-3">
            <span className="font-semibold text-gis-textMuted uppercase text-[10px] tracking-wider">
              Building Color
            </span>
            <div className="flex rounded bg-gis-surface p-0.5 border border-gis-border">
              <button
                onClick={() => onChangeVisualizationMode('normal')}
                className={`px-2 py-0.5 rounded text-[11px] font-medium ${
                  visualizationMode === 'normal'
                    ? 'bg-gis-accent text-white'
                    : 'text-gis-textMuted hover:text-white'
                }`}
              >
                Normal
              </button>
              <button
                onClick={() => onChangeVisualizationMode('confidence')}
                className={`px-2 py-0.5 rounded text-[11px] font-medium ${
                  visualizationMode === 'confidence'
                    ? 'bg-gis-cyan text-white'
                    : 'text-gis-textMuted hover:text-white'
                }`}
              >
                Confidence
              </button>
            </div>
          </div>
        </div>

        {/* Recenter Button */}
        <button
          onClick={handleRecenter}
          className="self-start bg-gis-card/90 border border-gis-border hover:bg-gis-surface backdrop-blur-md p-2 rounded-lg text-white shadow-lg transition-colors flex items-center space-x-1.5 text-xs"
          title="Recenter view"
        >
          <Compass className="w-4 h-4 text-gis-cyan" />
          <span>Recenter Map</span>
        </button>
      </div>
    </div>
  );
};

export default MapView;
