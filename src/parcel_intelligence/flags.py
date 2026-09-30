"""
src/parcel_intelligence/flags.py

Phase 14A — Uncertainty Flag Generator for SIH26012

Generates specific, actionable uncertainty flags for each parcel candidate
that indicate areas requiring human professional review.

Flag Categories:
  - NEEDS_BOUNDARY_REVIEW: Geometry is irregular or was repaired
  - WEAK_BUILDING_EVIDENCE: Low containment or implausible coverage
  - WEAK_ROAD_EVIDENCE: No road frontage or landlocked
  - NEEDS_TOPOLOGY_REVIEW: Flagged geometry or duplicate
  - HIGH_ELONGATION: Aspect ratio suggests boundary misalignment
  - OVERSIZED_PARCEL: Area exceeds plausible urban parcel range
  - UNDERSIZED_PARCEL: Area below minimum viable parcel size
  - MULTI_SOURCE_CONFLICT: Road evidence from single source only
  - LOW_CONFIDENCE: Overall confidence below threshold

SCIENTIFIC INTEGRITY:
  - Flags indicate areas for professional review, NOT defects.
  - No flags claim legal cadastral certainty.
"""

from typing import Dict, Any, List, Tuple


# ─── Flag Definitions ───
FLAG_DEFINITIONS = {
    "NEEDS_BOUNDARY_REVIEW": {
        "severity": "WARNING",
        "description": "Parcel boundary has irregular geometry (low compactness or was repaired). Recommend human review of boundary alignment.",
    },
    "WEAK_BUILDING_EVIDENCE": {
        "severity": "WARNING",
        "description": "Building footprint is not fully contained or coverage ratio is outside plausible range. Building-parcel relationship needs verification.",
    },
    "WEAK_ROAD_EVIDENCE": {
        "severity": "WARNING",
        "description": "Parcel has no direct road frontage or is landlocked. Access corridor evidence is weak.",
    },
    "NEEDS_TOPOLOGY_REVIEW": {
        "severity": "CRITICAL",
        "description": "Geometry has topological issues (flagged invalid or duplicate). Manual inspection required.",
    },
    "HIGH_ELONGATION": {
        "severity": "INFO",
        "description": "Parcel shape is highly elongated (aspect ratio > 5.0). May indicate boundary misalignment or sliver artifact.",
    },
    "OVERSIZED_PARCEL": {
        "severity": "INFO",
        "description": "Parcel area exceeds 2000 m². May represent aggregated compound or open land rather than individual parcel.",
    },
    "UNDERSIZED_PARCEL": {
        "severity": "INFO",
        "description": "Parcel area is below 20 m². May represent sliver artifact or partial building footprint.",
    },
    "MULTI_SOURCE_CONFLICT": {
        "severity": "INFO",
        "description": "Nearest road evidence comes from a single extraction source (not mutually confirmed). Consider verifying road presence.",
    },
    "LOW_CONFIDENCE": {
        "severity": "WARNING",
        "description": "Overall AI confidence score is below 0.50. Multiple evidence dimensions are weak.",
    },
}


class UncertaintyFlagger:
    """
    Generates actionable uncertainty flags for each parcel candidate
    based on evidence records and confidence results.
    """

    def __init__(
        self,
        compactness_threshold: float = 0.25,
        containment_threshold: float = 0.80,
        elongation_threshold: float = 5.0,
        area_max_m2: float = 2000.0,
        area_min_m2: float = 20.0,
        confidence_threshold: float = 0.50,
    ):
        self.compactness_threshold = compactness_threshold
        self.containment_threshold = containment_threshold
        self.elongation_threshold = elongation_threshold
        self.area_max_m2 = area_max_m2
        self.area_min_m2 = area_min_m2
        self.confidence_threshold = confidence_threshold

    def flag_single(
        self,
        evidence: Dict[str, Any],
        confidence_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generates flags for a single parcel candidate.

        Returns:
            List of flag dictionaries with flag_id, severity, reason, and definition.
        """
        flags = []

        geom = evidence.get("geometry", {})
        bldg = evidence.get("building", {})
        road = evidence.get("road_access", {})
        topo = evidence.get("topology", {})

        confidence = confidence_result.get("confidence", 0.0)

        # ── NEEDS_BOUNDARY_REVIEW ──
        pp = geom.get("polsby_popper_compactness", 0.0)
        was_repaired = topo.get("was_repaired", False)
        if pp < self.compactness_threshold or was_repaired:
            reason_parts = []
            if pp < self.compactness_threshold:
                reason_parts.append(f"compactness={pp:.3f} < {self.compactness_threshold}")
            if was_repaired:
                reason_parts.append("geometry was auto-repaired")
            flags.append(self._make_flag(
                "NEEDS_BOUNDARY_REVIEW",
                "; ".join(reason_parts),
            ))

        # ── WEAK_BUILDING_EVIDENCE ──
        containment = bldg.get("containment_ratio", 0.0)
        coverage_ok = bldg.get("coverage_plausible", True)
        if containment < self.containment_threshold or not coverage_ok:
            reason_parts = []
            if containment < self.containment_threshold:
                reason_parts.append(f"containment={containment:.3f} < {self.containment_threshold}")
            if not coverage_ok:
                cov = bldg.get("coverage_ratio", 0.0)
                reason_parts.append(f"coverage_ratio={cov:.3f} outside [0.05, 0.80]")
            flags.append(self._make_flag(
                "WEAK_BUILDING_EVIDENCE",
                "; ".join(reason_parts),
            ))

        # ── WEAK_ROAD_EVIDENCE ──
        frontage = road.get("road_frontage_m", 0.0)
        road_dist = road.get("road_distance_m", None)
        touches = road.get("road_touches", False)
        if not touches and (road_dist is None or road_dist > 10.0):
            flags.append(self._make_flag(
                "WEAK_ROAD_EVIDENCE",
                f"road_distance={road_dist}m, frontage={frontage:.1f}m, no direct road contact",
            ))
        elif frontage < 2.0 and touches:
            flags.append(self._make_flag(
                "WEAK_ROAD_EVIDENCE",
                f"very narrow frontage={frontage:.1f}m despite corridor contact",
            ))

        # ── NEEDS_TOPOLOGY_REVIEW ──
        is_flagged = topo.get("is_flagged", False)
        if is_flagged:
            flags.append(self._make_flag(
                "NEEDS_TOPOLOGY_REVIEW",
                f"topology_status={topo.get('topology_status', 'unknown')}",
            ))

        # ── HIGH_ELONGATION ──
        aspect = geom.get("aspect_ratio", 1.0)
        if aspect > self.elongation_threshold:
            flags.append(self._make_flag(
                "HIGH_ELONGATION",
                f"aspect_ratio={aspect:.2f} > {self.elongation_threshold}",
            ))

        # ── OVERSIZED_PARCEL ──
        area = geom.get("area_m2", 0.0)
        if area > self.area_max_m2:
            flags.append(self._make_flag(
                "OVERSIZED_PARCEL",
                f"area={area:.1f}m² > {self.area_max_m2}m²",
            ))

        # ── UNDERSIZED_PARCEL ──
        if area < self.area_min_m2:
            flags.append(self._make_flag(
                "UNDERSIZED_PARCEL",
                f"area={area:.1f}m² < {self.area_min_m2}m²",
            ))

        # ── MULTI_SOURCE_CONFLICT ──
        source = road.get("nearest_road_source", "unknown")
        if source in ("classical", "deep_learning"):
            flags.append(self._make_flag(
                "MULTI_SOURCE_CONFLICT",
                f"nearest road from single source: {source}",
            ))

        # ── LOW_CONFIDENCE ──
        if confidence < self.confidence_threshold:
            flags.append(self._make_flag(
                "LOW_CONFIDENCE",
                f"confidence={confidence:.4f} < {self.confidence_threshold}",
            ))

        return flags

    def flag_batch(
        self,
        evidence_list: List[Dict[str, Any]],
        confidence_results: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """Flags all candidates in batch."""
        return [
            self.flag_single(ev, cr)
            for ev, cr in zip(evidence_list, confidence_results)
        ]

    def compute_flag_summary(
        self,
        all_flags: List[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Computes population-level flag distribution statistics.
        """
        n = len(all_flags)
        total_flags = sum(len(fl) for fl in all_flags)
        parcels_with_flags = sum(1 for fl in all_flags if len(fl) > 0)

        flag_type_counts = {}
        severity_counts = {"CRITICAL": 0, "WARNING": 0, "INFO": 0}

        for fl_list in all_flags:
            for flag in fl_list:
                fid = flag["flag_id"]
                flag_type_counts[fid] = flag_type_counts.get(fid, 0) + 1
                sev = flag.get("severity", "INFO")
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return {
            "total_candidates": n,
            "candidates_with_flags": parcels_with_flags,
            "candidates_without_flags": n - parcels_with_flags,
            "flag_rate_pct": round(100.0 * parcels_with_flags / max(1, n), 2),
            "total_flags_raised": total_flags,
            "mean_flags_per_candidate": round(total_flags / max(1, n), 2),
            "flag_type_distribution": dict(sorted(flag_type_counts.items(), key=lambda x: -x[1])),
            "severity_distribution": severity_counts,
        }

    def _make_flag(self, flag_id: str, reason: str) -> Dict[str, Any]:
        """Constructs a structured flag record."""
        defn = FLAG_DEFINITIONS.get(flag_id, {})
        return {
            "flag_id": flag_id,
            "severity": defn.get("severity", "INFO"),
            "reason": reason,
            "definition": defn.get("description", ""),
        }
