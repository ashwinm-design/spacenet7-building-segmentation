"""
src/parcel_intelligence/evidence.py

Phase 14A — Evidence Assembler for SIH26012

Gathers all measurable AI-inferred evidence for each parcel candidate
from existing pipeline outputs:
  1. Geometry quality signals (area, compactness, aspect ratio)
  2. Building containment evidence
  3. Road access / frontage evidence
  4. Topological integrity signals
  5. Neighborhood context signals

SCIENTIFIC INTEGRITY:
  - Does NOT fabricate cadastral ground truth.
  - Does NOT claim legal property boundaries.
  - All evidence is from AI-inferred, measurable attributes.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union


def _aspect_ratio(geom: Any) -> float:
    """
    Computes oriented bounding box aspect ratio.
    Returns value >= 1.0 (length / width).
    """
    if geom is None or geom.is_empty:
        return 1.0
    try:
        obb = geom.minimum_rotated_rectangle
        coords = list(obb.exterior.coords)
        d1 = np.sqrt((coords[0][0] - coords[1][0])**2 + (coords[0][1] - coords[1][1])**2)
        d2 = np.sqrt((coords[1][0] - coords[2][0])**2 + (coords[1][1] - coords[2][1])**2)
        if min(d1, d2) < 1e-6:
            return 10.0
        return max(d1, d2) / min(d1, d2)
    except Exception:
        return 1.0


def _convexity(geom: Any) -> float:
    """
    Ratio of polygon area to its convex hull area.
    Values near 1.0 indicate convex shapes (typical parcels).
    """
    if geom is None or geom.is_empty:
        return 0.0
    hull_area = geom.convex_hull.area
    if hull_area < 1e-6:
        return 0.0
    return float(np.clip(geom.area / hull_area, 0.0, 1.0))


class EvidenceAssembler:
    """
    Gathers structured evidence records for each parcel candidate.
    Each evidence record is a dictionary of measurable attributes
    organized into evidence categories.
    """

    def __init__(
        self,
        buildings_gdf: gpd.GeoDataFrame,
        road_gdf: Optional[gpd.GeoDataFrame] = None,
        road_edges_gdf: Optional[gpd.GeoDataFrame] = None,
    ):
        self.buildings_gdf = buildings_gdf
        self.road_gdf = road_gdf
        self.road_edges_gdf = road_edges_gdf

        # Pre-compute building lookup
        self._bldg_lookup = {}
        for i, row in buildings_gdf.iterrows():
            bid = row.get("id", i)
            self._bldg_lookup[bid] = row.geometry

        # Pre-compute road union
        self._road_union = None
        if road_gdf is not None and len(road_gdf) > 0:
            self._road_union = unary_union(road_gdf.geometry.tolist())
            if self._road_union.is_empty:
                self._road_union = None

    def assemble_evidence(
        self,
        parcels_gdf: gpd.GeoDataFrame,
    ) -> List[Dict[str, Any]]:
        """
        Assembles comprehensive evidence for each candidate parcel.

        Returns:
            List of evidence dictionaries, one per parcel.
        """
        evidence_list = []

        for idx, row in parcels_gdf.iterrows():
            geom = row.geometry
            evidence = self._build_single_evidence(row, geom, idx)
            evidence_list.append(evidence)

        return evidence_list

    def _build_single_evidence(
        self, row: Any, geom: Any, idx: int
    ) -> Dict[str, Any]:
        """Builds evidence record for a single parcel candidate."""

        parcel_id = row.get("parcel_id", idx)
        building_id = row.get("building_id", idx)
        b_geom = self._bldg_lookup.get(building_id, None)

        # ── 1. Geometry Quality Evidence ──
        area = float(geom.area) if geom is not None and not geom.is_empty else 0.0
        perimeter = float(geom.length) if geom is not None and not geom.is_empty else 0.0

        polsby_popper = 0.0
        if perimeter > 0 and area > 0:
            polsby_popper = float(np.clip(
                (4.0 * np.pi * area) / (perimeter ** 2), 0.0, 1.0
            ))

        aspect = _aspect_ratio(geom)
        convexity = _convexity(geom)

        # Area plausibility: typical urban parcels 20-2000 m²
        area_plausible = 20.0 <= area <= 2000.0

        geometry_evidence = {
            "area_m2": round(area, 2),
            "perimeter_m": round(perimeter, 2),
            "polsby_popper_compactness": round(polsby_popper, 4),
            "aspect_ratio": round(aspect, 3),
            "convexity": round(convexity, 4),
            "area_plausible": area_plausible,
        }

        # ── 2. Building Containment Evidence ──
        containment_ratio = 0.0
        building_area = 0.0
        coverage_ratio = 0.0
        building_fully_contained = False

        if b_geom is not None and geom is not None and not geom.is_empty:
            building_area = float(b_geom.area)
            if geom.contains(b_geom):
                containment_ratio = 1.0
                building_fully_contained = True
            else:
                inter_area = geom.intersection(b_geom).area
                containment_ratio = float(np.clip(
                    inter_area / max(1e-4, building_area), 0.0, 1.0
                ))

            coverage_ratio = building_area / max(1e-4, area)

        building_evidence = {
            "building_area_m2": round(building_area, 2),
            "containment_ratio": round(containment_ratio, 4),
            "building_fully_contained": building_fully_contained,
            "coverage_ratio": round(coverage_ratio, 4),
            "coverage_plausible": 0.05 <= coverage_ratio <= 0.80,
        }

        # ── 3. Road Access Evidence ──
        road_distance_m = row.get("road_distance_m", None)
        road_frontage_m = row.get("road_frontage_m", 0.0)
        road_touches = row.get("road_touches", False)
        road_evidence_score = row.get("road_evidence_score", None)
        road_evidence_text = row.get("road_evidence", "unavailable")

        # Compute nearest road edge source evidence if available
        nearest_edge_source = "unknown"
        nearest_edge_confidence = 0.0
        if self.road_edges_gdf is not None and len(self.road_edges_gdf) > 0 and geom is not None:
            try:
                centroid = geom.centroid
                dists = self.road_edges_gdf.geometry.distance(centroid)
                nearest_idx = dists.idxmin()
                nearest_edge = self.road_edges_gdf.loc[nearest_idx]
                nearest_edge_source = str(nearest_edge.get("source_evidence", "unknown"))
                nearest_edge_confidence = float(nearest_edge.get("confidence", 0.0))
            except Exception:
                pass

        road_access_evidence = {
            "road_distance_m": float(road_distance_m) if road_distance_m is not None else None,
            "road_frontage_m": round(float(road_frontage_m), 2),
            "road_touches": bool(road_touches),
            "road_evidence_score": float(road_evidence_score) if road_evidence_score is not None else None,
            "road_evidence_text": str(road_evidence_text),
            "nearest_road_source": nearest_edge_source,
            "nearest_road_confidence": round(nearest_edge_confidence, 3),
        }

        # ── 4. Topological Integrity Evidence ──
        topology_status = str(row.get("topology_status", "unknown"))

        topology_evidence = {
            "topology_status": topology_status,
            "is_valid": topology_status in ("valid", "repaired_valid", "repaired_single_part"),
            "was_repaired": "repaired" in topology_status,
            "is_flagged": "flagged" in topology_status,
        }

        # ── 5. Neighborhood Context ──
        cluster_id = row.get("cluster_id", -1)

        neighborhood_evidence = {
            "cluster_id": int(cluster_id) if cluster_id is not None else -1,
        }

        return {
            "parcel_id": int(parcel_id),
            "building_id": int(building_id) if building_id is not None else None,
            "geometry": geometry_evidence,
            "building": building_evidence,
            "road_access": road_access_evidence,
            "topology": topology_evidence,
            "neighborhood": neighborhood_evidence,
        }
