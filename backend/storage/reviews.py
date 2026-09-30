"""
backend/storage/reviews.py

Persistent Human-in-the-Loop review storage and geometric recalculation service
for preliminary parcel candidates (Phase 14B).
"""

import os
import json
import math
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

import numpy as np
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.validation import make_valid
import pyproj

API_STORAGE_ROOT = Path("results/prototype/api")
FALLBACK_AUDITED_GEOJSON = Path("results/experiments/parcel_intelligence/parcels_audited.geojson")
FALLBACK_ROADS_GEOJSON = Path("results/experiments/road_hybrid/roads_hybrid.geojson")


def get_geometry_hash(geom: Dict[str, Any]) -> str:
    """Returns a deterministic short SHA-256 hash of GeoJSON geometry coordinates."""
    try:
        coords_str = json.dumps(geom.get("coordinates", []), sort_keys=True)
        return hashlib.sha256(coords_str.encode("utf-8")).hexdigest()[:12]
    except Exception:
        return "unknown"


def compute_polsby_popper_compactness(area: float, perimeter: float) -> float:
    """Computes Polsby-Popper compactness: 4 * pi * Area / Perimeter^2."""
    if perimeter <= 0 or area <= 0:
        return 0.0
    val = (4.0 * math.pi * area) / (perimeter ** 2)
    return float(np.clip(val, 0.0, 1.0))


class ReviewStorageManager:
    """
    Manages parcel review state, history audit trail, geometry validation,
    and attribute recalculation.
    """

    def __init__(self, root_dir: Path = API_STORAGE_ROOT):
        self.root_dir = root_dir
        self._cached_roads: Dict[str, Any] = {}
        self._cached_buildings: Dict[str, Dict[int, Any]] = {}

    def get_analysis_dir(self, analysis_id: str) -> Path:
        return self.root_dir / analysis_id

    def get_reviews_path(self, analysis_id: str) -> Path:
        return self.get_analysis_dir(analysis_id) / "parcel_reviews.json"

    def get_audit_path(self, analysis_id: str) -> Path:
        return self.get_analysis_dir(analysis_id) / "parcel_reviews_audit.json"

    def get_reviewed_geojson_path(self, analysis_id: str) -> Path:
        return self.get_analysis_dir(analysis_id) / "reviewed_parcels.geojson"

    def get_review_summary_path(self, analysis_id: str) -> Path:
        return self.get_analysis_dir(analysis_id) / "review_summary.json"

    def _ensure_initialized(self, analysis_id: str) -> Dict[str, Any]:
        """
        Ensures review store exists for this analysis_id.
        If parcel_reviews.json does not exist, initializes it from parcels_audited.geojson
        or parcels_hybrid.geojson with all 254 candidates defaulted to 'unreviewed'.
        """
        rev_path = self.get_reviews_path(analysis_id)
        if rev_path.is_file():
            try:
                with open(rev_path, "r", encoding="utf-8") as f:
                    reviews = json.load(f)
                    
                # Check if items need metric enrichment
                needs_enrichment = False
                sample_item = next(iter(reviews.values()), {}) if reviews else {}
                if "area_m2" not in sample_item or "conf_building_evidence" not in sample_item:
                    needs_enrichment = True
                    
                if not needs_enrichment:
                    return reviews
                
                # Enrich with audited properties
                audited_map = {}
                if FALLBACK_AUDITED_GEOJSON.is_file():
                    try:
                        with open(FALLBACK_AUDITED_GEOJSON, "r", encoding="utf-8") as af:
                            aud_data = json.load(af)
                            for feat in aud_data.get("features", []):
                                pid = feat["properties"].get("parcel_id")
                                if pid is not None:
                                    audited_map[int(pid)] = feat["properties"]
                    except Exception:
                        pass
                
                for cid_str, item in reviews.items():
                    cid = int(cid_str)
                    aud_props = audited_map.get(cid, {})
                    if "area_m2" not in item:
                        item["area_m2"] = round(float(aud_props.get("parcel_area", 0.0)), 2)
                    if "building_coverage" not in item:
                        item["building_coverage"] = round(float(aud_props.get("coverage_ratio", 0.0)), 4)
                    if "compactness" not in item:
                        item["compactness"] = round(float(aud_props.get("compactness", 0.0)), 4)
                    if "road_frontage" not in item:
                        item["road_frontage"] = round(float(aud_props.get("road_frontage_m", 0.0)), 2)
                    if "nearest_road_distance" not in item:
                        item["nearest_road_distance"] = round(float(aud_props.get("road_distance_m", 0.0)), 2)
                    if "conf_building_evidence" not in item:
                        item["conf_building_evidence"] = round(float(aud_props.get("conf_building_evidence", 0.0)), 4)
                    if "conf_road_access" not in item:
                        item["conf_road_access"] = round(float(aud_props.get("conf_road_access", 0.0)), 4)
                    if "conf_geometry_quality" not in item:
                        item["conf_geometry_quality"] = round(float(aud_props.get("conf_geometry_quality", 0.0)), 4)
                    if "conf_topological_plausibility" not in item:
                        item["conf_topological_plausibility"] = round(float(aud_props.get("conf_topological_plausibility", 1.0)), 4)
                    if "recommendation" not in item:
                        item["recommendation"] = aud_props.get("recommendation", "")
                
                with open(rev_path, "w", encoding="utf-8") as f:
                    json.dump(reviews, f, indent=2)
                return reviews
            except Exception:
                pass

        # Need to initialize from source parcels
        analysis_dir = self.get_analysis_dir(analysis_id)
        analysis_dir.mkdir(parents=True, exist_ok=True)

        source_geojson = analysis_dir / "parcels_hybrid.geojson"
        if not source_geojson.is_file():
            source_geojson = FALLBACK_AUDITED_GEOJSON
        if not source_geojson.is_file():
            source_geojson = Path("results/experiments/road_hybrid/candidate_parcels_hybrid_constrained.geojson")

        if not source_geojson.is_file():
            return {}

        with open(source_geojson, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Also load audited attributes if available
        audited_map = {}
        local_audited = analysis_dir / "parcels_audited.geojson"
        target_audited = local_audited if local_audited.is_file() else FALLBACK_AUDITED_GEOJSON
        if target_audited.is_file():
            try:
                with open(target_audited, "r", encoding="utf-8") as f:
                    aud_data = json.load(f)
                    for feat in aud_data.get("features", []):
                        pid = feat["properties"].get("parcel_id")
                        if pid is not None:
                            audited_map[int(pid)] = feat["properties"]
            except Exception:
                pass

        reviews: Dict[str, Any] = {}
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            cid = props.get("parcel_id") or props.get("candidate_id") or props.get("building_id")
            if cid is None:
                continue
            cid = int(cid)

            aud_props = audited_map.get(cid, props)
            confidence = float(aud_props.get("confidence", props.get("candidate_score", 0.85)))
            confidence_band = aud_props.get("confidence_band", "HIGH" if confidence >= 0.75 else "MODERATE")
            flags = aud_props.get("flags", "")

            # Required review state data model
            geom = feat.get("geometry", {})
            reviews[str(cid)] = {
                "candidate_id": cid,
                "analysis_id": analysis_id,
                "review_status": "unreviewed",
                "reviewer_note": None,
                "confidence": round(confidence, 4),
                "confidence_band": confidence_band,
                "flags": flags,
                "original_geometry": geom,
                "current_geometry": geom,
                "geometry_modified": False,
                "review_timestamp": None,
                # Additional derived metrics for rich inspection panel
                "area_m2": round(float(aud_props.get("parcel_area", 0.0)), 2),
                "building_coverage": round(float(aud_props.get("coverage_ratio", 0.0)), 4),
                "compactness": round(float(aud_props.get("compactness", 0.0)), 4),
                "road_frontage": round(float(aud_props.get("road_frontage_m", 0.0)), 2),
                "nearest_road_distance": round(float(aud_props.get("road_distance_m", 0.0)), 2),
                "conf_building_evidence": round(float(aud_props.get("conf_building_evidence", 0.0)), 4),
                "conf_road_access": round(float(aud_props.get("conf_road_access", 0.0)), 4),
                "conf_geometry_quality": round(float(aud_props.get("conf_geometry_quality", 0.0)), 4),
                "conf_topological_plausibility": round(float(aud_props.get("conf_topological_plausibility", 1.0)), 4),
                "recommendation": aud_props.get("recommendation", "AI Candidate"),
            }

        with open(rev_path, "w", encoding="utf-8") as f:
            json.dump(reviews, f, indent=2)

        return reviews

    def get_all_reviews(self, analysis_id: str) -> Dict[str, Any]:
        """Returns all parcel reviews for analysis_id."""
        return self._ensure_initialized(analysis_id)

    def get_review(self, analysis_id: str, candidate_id: int) -> Optional[Dict[str, Any]]:
        """Returns review state for a single candidate."""
        reviews = self.get_all_reviews(analysis_id)
        return reviews.get(str(candidate_id))

    def get_summary(self, analysis_id: str) -> Dict[str, int]:
        """Calculates total, unreviewed, accepted, edited, needs_review, rejected."""
        reviews = self.get_all_reviews(analysis_id)
        counts = {
            "total": len(reviews),
            "unreviewed": 0,
            "accepted": 0,
            "edited": 0,
            "needs_review": 0,
            "rejected": 0,
        }
        for item in reviews.values():
            st = item.get("review_status", "unreviewed")
            if st in counts:
                counts[st] += 1
            else:
                counts["unreviewed"] += 1
        return counts

    def _load_roads_union(self, analysis_id: str) -> Optional[Any]:
        """Loads and caches merged road corridor geometry for road distance/frontage calculation."""
        if analysis_id in self._cached_roads:
            return self._cached_roads[analysis_id]

        p = self.get_analysis_dir(analysis_id) / "roads_hybrid.geojson"
        if not p.is_file():
            p = FALLBACK_ROADS_GEOJSON

        if not p.is_file():
            return None

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            geoms = []
            for feat in data.get("features", []):
                g = shape(feat["geometry"])
                if g.is_valid and not g.is_empty:
                    geoms.append(g)
            if geoms:
                from shapely.ops import unary_union
                road_union = unary_union(geoms)
                self._cached_roads[analysis_id] = road_union
                return road_union
        except Exception:
            pass
        return None

    def _load_building_geometry(self, analysis_id: str, building_id: int) -> Tuple[Optional[Any], float]:
        """Loads building footprint geometry and area for building coverage calculation."""
        if analysis_id not in self._cached_buildings:
            bldgs: Dict[int, Any] = {}
            p = self.get_analysis_dir(analysis_id) / "buildings.geojson"
            if p.is_file():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for feat in data.get("features", []):
                        bid = feat.get("properties", {}).get("id")
                        if bid is not None:
                            bldgs[int(bid)] = shape(feat["geometry"])
                except Exception:
                    pass
            self._cached_buildings[analysis_id] = bldgs

        bldg_geom = self._cached_buildings.get(analysis_id, {}).get(building_id)
        bldg_area = bldg_geom.area if bldg_geom is not None and bldg_geom.is_valid else 0.0
        return bldg_geom, bldg_area

    def validate_and_repair_geometry(
        self,
        geometry_dict: Dict[str, Any],
        expected_crs: str = "EPSG:32732"
    ) -> Tuple[Dict[str, Any], Any]:
        """
        Validates GeoJSON geometry:
        - Must be Polygon or MultiPolygon
        - Coordinates must not contain NaN or Inf
        - Checks validity, repairs safely using shapely.make_valid
        - Handles WGS84 input by converting to projected CRS (EPSG:32732)
        Returns: (geojson_dict_in_projected_crs, shapely_geom)
        """
        if not isinstance(geometry_dict, dict):
            raise ValueError("Geometry must be a valid GeoJSON object.")

        geom_type = geometry_dict.get("type")
        if geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"Geometry type '{geom_type}' is not supported. Must be Polygon or MultiPolygon.")

        coords = geometry_dict.get("coordinates")
        if not coords or not isinstance(coords, (list, tuple)):
            raise ValueError("Geometry coordinates must be a non-empty list.")

        # Check for NaN / Inf in coordinates recursively
        def check_coords(val):
            if isinstance(val, (int, float)):
                if math.isnan(val) or math.isinf(val):
                    raise ValueError("Geometry coordinates contain invalid NaN or Inf values.")
            elif isinstance(val, (list, tuple)):
                for sub in val:
                    check_coords(sub)

        check_coords(coords)

        # Convert to shapely geometry
        try:
            s_geom = shape(geometry_dict)
        except Exception as e:
            raise ValueError(f"Invalid GeoJSON geometry representation: {str(e)}")

        if s_geom.is_empty:
            raise ValueError("Geometry cannot be empty.")

        # Check coordinate range to determine if coordinates are in WGS84 (lon/lat)
        # If coordinates are in degrees [-180, 180], reproject to EPSG:32732
        bounds = s_geom.bounds  # (minx, miny, maxx, maxy)
        if -180.0 <= bounds[0] <= 180.0 and -90.0 <= bounds[1] <= 90.0:
            # Coordinates are in WGS84 degrees
            try:
                from shapely.ops import transform
                proj_transformer = pyproj.Transformer.from_crs("EPSG:4326", expected_crs, always_xy=True).transform
                s_geom = transform(proj_transformer, s_geom)
            except Exception as e:
                raise ValueError(f"Failed to reproject coordinates from WGS84 to {expected_crs}: {str(e)}")

        # Validate and repair if needed
        if not s_geom.is_valid:
            try:
                repaired = make_valid(s_geom)
                if isinstance(repaired, (Polygon, MultiPolygon)) and not repaired.is_empty:
                    s_geom = repaired
                else:
                    # Filter for Polygon/MultiPolygon components if GeometryCollection returned
                    polys = [g for g in getattr(repaired, "geoms", []) if isinstance(g, (Polygon, MultiPolygon))]
                    if polys:
                        from shapely.ops import unary_union
                        s_geom = unary_union(polys)
                    else:
                        raise ValueError("Repaired geometry does not produce a valid Polygon.")
            except Exception as e:
                raise ValueError(f"Geometry is invalid and could not be safely repaired: {str(e)}")

        if s_geom.area <= 0.0:
            raise ValueError("Geometry area must be greater than zero.")

        return mapping(s_geom), s_geom

    def recalculate_attributes(
        self,
        analysis_id: str,
        candidate_id: int,
        s_geom: Any,
    ) -> Dict[str, float]:
        """
        Recalculates geometry-dependent attributes for edited parcel:
        - area_m2
        - building_coverage
        - compactness
        - road_frontage
        - nearest_road_distance
        """
        area_m2 = float(s_geom.area)
        perimeter = float(s_geom.length)
        compactness = compute_polsby_popper_compactness(area_m2, perimeter)

        # Building coverage
        bldg_geom, bldg_area = self._load_building_geometry(analysis_id, candidate_id)
        if bldg_geom is not None and not bldg_geom.is_empty:
            if s_geom.contains(bldg_geom):
                bldg_actual_area = bldg_area
            else:
                try:
                    bldg_actual_area = s_geom.intersection(bldg_geom).area
                except Exception:
                    bldg_actual_area = bldg_area
        else:
            bldg_actual_area = bldg_area

        building_coverage = float(np.clip(bldg_actual_area / max(1e-4, area_m2), 0.0, 1.0))

        # Road distance and frontage
        road_union = self._load_roads_union(analysis_id)
        if road_union is not None and not road_union.is_empty:
            try:
                dist = float(s_geom.distance(road_union))
            except Exception:
                dist = 0.0
            nearest_road_distance = max(0.0, dist)

            road_frontage = 0.0
            if dist < 0.25:
                nearest_road_distance = 0.0
                try:
                    shared_line = s_geom.boundary.intersection(road_union.buffer(0.1))
                    road_frontage = float(shared_line.length)
                except Exception:
                    road_frontage = 0.0
        else:
            nearest_road_distance = 0.0
            road_frontage = 0.0

        return {
            "area_m2": round(area_m2, 2),
            "building_coverage": round(building_coverage, 4),
            "compactness": round(compactness, 4),
            "road_frontage": round(road_frontage, 2),
            "nearest_road_distance": round(nearest_road_distance, 2),
        }

    def update_review(
        self,
        analysis_id: str,
        candidate_id: int,
        review_status: str,
        reviewer_note: Optional[str] = None,
        new_geometry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Updates review status, reviewer note, and optional edited geometry.
        Audits change history and updates review store.
        """
        allowed_statuses = {"unreviewed", "accepted", "edited", "rejected", "needs_review"}
        if review_status not in allowed_statuses:
            raise ValueError(f"Invalid review_status '{review_status}'. Allowed: {sorted(allowed_statuses)}")

        reviews = self.get_all_reviews(analysis_id)
        cid_str = str(candidate_id)
        if cid_str not in reviews:
            raise KeyError(f"Candidate ID #{candidate_id} not found in analysis '{analysis_id}'.")

        item = reviews[cid_str]
        old_status = item.get("review_status", "unreviewed")
        old_geom = item.get("current_geometry", item.get("original_geometry"))
        old_hash = get_geometry_hash(old_geom)

        timestamp = datetime.now(timezone.utc).isoformat()
        geometry_modified = item.get("geometry_modified", False)

        # Handle geometry update if provided
        if new_geometry is not None:
            validated_geom_dict, s_geom = self.validate_and_repair_geometry(new_geometry)
            item["current_geometry"] = validated_geom_dict
            new_hash = get_geometry_hash(validated_geom_dict)

            # Check if geometry actually differed from original
            orig_hash = get_geometry_hash(item["original_geometry"])
            geometry_modified = (new_hash != orig_hash)
            item["geometry_modified"] = geometry_modified

            # Recalculate geometry-dependent metrics
            recalc = self.recalculate_attributes(analysis_id, candidate_id, s_geom)
            item.update(recalc)

            if review_status == "unreviewed":
                review_status = "edited"
        else:
            new_hash = old_hash

        item["review_status"] = review_status
        item["reviewer_note"] = reviewer_note
        item["review_timestamp"] = timestamp

        # Persist reviews
        rev_path = self.get_reviews_path(analysis_id)
        with open(rev_path, "w", encoding="utf-8") as f:
            json.dump(reviews, f, indent=2)

        # Append to audit trail
        self._record_audit_entry(
            analysis_id=analysis_id,
            candidate_id=candidate_id,
            old_status=old_status,
            new_status=review_status,
            old_geometry_hash=old_hash,
            new_geometry_hash=new_hash,
            reviewer_note=reviewer_note,
            timestamp=timestamp,
        )

        return item

    def reset_review_geometry(self, analysis_id: str, candidate_id: int) -> Dict[str, Any]:
        """
        Resets parcel current_geometry back to original_geometry and recalculates original metrics.
        """
        reviews = self.get_all_reviews(analysis_id)
        cid_str = str(candidate_id)
        if cid_str not in reviews:
            raise KeyError(f"Candidate ID #{candidate_id} not found in analysis '{analysis_id}'.")

        item = reviews[cid_str]
        old_status = item.get("review_status", "unreviewed")
        old_geom = item.get("current_geometry", item.get("original_geometry"))
        old_hash = get_geometry_hash(old_geom)

        # Revert geometry to original
        orig_geom = item["original_geometry"]
        item["current_geometry"] = orig_geom
        item["geometry_modified"] = False
        new_hash = get_geometry_hash(orig_geom)

        # Recalculate metrics for original geometry
        try:
            s_geom = shape(orig_geom)
            recalc = self.recalculate_attributes(analysis_id, candidate_id, s_geom)
            item.update(recalc)
        except Exception:
            pass

        timestamp = datetime.now(timezone.utc).isoformat()
        # If it was edited, set to needs_review or unreviewed
        if item.get("review_status") == "edited":
            item["review_status"] = "unreviewed"

        item["review_timestamp"] = timestamp

        # Persist
        rev_path = self.get_reviews_path(analysis_id)
        with open(rev_path, "w", encoding="utf-8") as f:
            json.dump(reviews, f, indent=2)

        # Audit
        self._record_audit_entry(
            analysis_id=analysis_id,
            candidate_id=candidate_id,
            old_status=old_status,
            new_status=item["review_status"],
            old_geometry_hash=old_hash,
            new_geometry_hash=new_hash,
            reviewer_note=f"Reset geometry to AI original: {item.get('reviewer_note') or ''}".strip(),
            timestamp=timestamp,
        )

        return item

    def _record_audit_entry(
        self,
        analysis_id: str,
        candidate_id: int,
        old_status: str,
        new_status: str,
        old_geometry_hash: str,
        new_geometry_hash: str,
        reviewer_note: Optional[str],
        timestamp: str,
    ):
        """Appends an entry to parcel_reviews_audit.json."""
        audit_path = self.get_audit_path(analysis_id)
        entries = []
        if audit_path.is_file():
            try:
                with open(audit_path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
            except Exception:
                entries = []

        entries.append({
            "candidate_id": candidate_id,
            "old_status": old_status,
            "new_status": new_status,
            "old_geometry_hash": old_geometry_hash,
            "new_geometry_hash": new_geometry_hash,
            "reviewer_note": reviewer_note,
            "timestamp": timestamp,
        })

        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)

    def export_reviewed_geojson(self, analysis_id: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Exports GIS-ready GeoJSON feature collection and review summary JSON.
        Features contain:
        candidate_id, review_status, confidence, confidence_band, flags,
        area_m2, building_coverage, road_frontage, nearest_road_distance,
        reviewer_note, geometry_modified.
        """
        reviews = self.get_all_reviews(analysis_id)
        summary = self.get_summary(analysis_id)

        features = []
        for cid_str, rev in sorted(reviews.items(), key=lambda x: int(x[0])):
            feat = {
                "type": "Feature",
                "id": rev["candidate_id"],
                "properties": {
                    "candidate_id": rev["candidate_id"],
                    "parcel_id": rev["candidate_id"],
                    "building_id": rev["candidate_id"],
                    "review_status": rev["review_status"],
                    "confidence": rev["confidence"],
                    "confidence_band": rev["confidence_band"],
                    "flags": rev["flags"] if isinstance(rev["flags"], str) else ",".join(rev["flags"]),
                    "area_m2": rev.get("area_m2", 0.0),
                    "parcel_area": rev.get("area_m2", 0.0),
                    "building_coverage": rev.get("building_coverage", 0.0),
                    "coverage_ratio": rev.get("building_coverage", 0.0),
                    "compactness": rev.get("compactness", 0.0),
                    "road_frontage": rev.get("road_frontage", 0.0),
                    "road_frontage_m": rev.get("road_frontage", 0.0),
                    "nearest_road_distance": rev.get("nearest_road_distance", 0.0),
                    "road_distance_m": rev.get("nearest_road_distance", 0.0),
                    "candidate_score": rev["confidence"],
                    "reviewer_note": rev.get("reviewer_note") or "",
                    "geometry_modified": rev.get("geometry_modified", False),
                    "review_timestamp": rev.get("review_timestamp") or "",
                    "conf_building_evidence": rev.get("conf_building_evidence", 0.0),
                    "conf_road_access": rev.get("conf_road_access", 0.0),
                    "conf_geometry_quality": rev.get("conf_geometry_quality", 0.0),
                    "conf_topological_plausibility": rev.get("conf_topological_plausibility", 1.0),
                    "recommendation": rev.get("recommendation", ""),
                    "disclaimer": (
                        "AI-generated parcel boundaries are preliminary spatial candidates and "
                        "require professional cadastral/survey verification. Review status does "
                        "not establish legal ownership or boundary rights."
                    ),
                },
                "geometry": rev.get("current_geometry", rev.get("original_geometry")),
            }
            features.append(feat)

        fc = {
            "type": "FeatureCollection",
            "name": f"reviewed_parcels_{analysis_id}",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:EPSG::32732"}
            },
            "features": features,
        }

        # Save to analysis directory
        reviewed_path = self.get_reviewed_geojson_path(analysis_id)
        with open(reviewed_path, "w", encoding="utf-8") as f:
            json.dump(fc, f, indent=2)

        summary_path = self.get_review_summary_path(analysis_id)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Also save to results/experiments/parcel_review/
        exp_dir = Path("results/experiments/parcel_review")
        exp_dir.mkdir(parents=True, exist_ok=True)
        with open(exp_dir / "reviewed_parcels.geojson", "w", encoding="utf-8") as f:
            json.dump(fc, f, indent=2)
        with open(exp_dir / "review_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return fc, summary


review_storage = ReviewStorageManager()
