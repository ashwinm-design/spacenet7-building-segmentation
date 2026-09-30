"""
src/parcel_intelligence/confidence.py

Phase 14A — Deterministic Parcel Confidence Engine for SIH26012

Computes a deterministic confidence score [0, 1] for each AI-generated
preliminary parcel candidate using a weighted composite of four
measurable evidence dimensions:

    1. Geometry Quality      (weight = 0.25)
    2. Building Evidence     (weight = 0.30)
    3. Road Access Evidence  (weight = 0.25)
    4. Topological Plausibility (weight = 0.20)

Each dimension produces a sub-score [0, 1] from measurable attributes.
The final confidence is the weighted sum, mapped to a categorical band:

    HIGH       >= 0.75
    MODERATE   >= 0.50
    LOW        >= 0.30
    VERY_LOW   < 0.30

SCIENTIFIC INTEGRITY:
  - Confidence measures AI inference quality, NOT legal cadastral certainty.
  - All inputs are derived from the existing pipeline outputs.
  - No ground truth data is required or fabricated.
"""

from typing import Dict, Any, List, Tuple
import numpy as np


# ─── Weight Configuration ───
WEIGHTS = {
    "geometry":   0.25,
    "building":   0.30,
    "road_access": 0.25,
    "topology":   0.20,
}

BAND_THRESHOLDS = [
    (0.75, "HIGH"),
    (0.50, "MODERATE"),
    (0.30, "LOW"),
    (0.00, "VERY_LOW"),
]


def _classify_band(score: float) -> str:
    """Maps a [0,1] confidence score to a categorical band."""
    for threshold, label in BAND_THRESHOLDS:
        if score >= threshold:
            return label
    return "VERY_LOW"


def _geometry_subscore(evidence: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Geometry quality sub-score from:
      - Polsby-Popper compactness (target: > 0.30)
      - Convexity (target: > 0.60)
      - Area plausibility (20-2000 m²)
      - Aspect ratio penalty (penalize > 5.0)
    """
    pp = evidence.get("polsby_popper_compactness", 0.0)
    convexity = evidence.get("convexity", 0.0)
    area_ok = evidence.get("area_plausible", False)
    aspect = evidence.get("aspect_ratio", 1.0)

    # Compactness component: scaled so 0.5+ → 1.0, 0.0 → 0.0
    compactness_score = float(np.clip(pp / 0.50, 0.0, 1.0))

    # Convexity component
    convexity_score = float(np.clip(convexity / 0.80, 0.0, 1.0))

    # Area plausibility
    area_score = 1.0 if area_ok else 0.3

    # Aspect ratio penalty: penalize elongated parcels
    if aspect <= 3.0:
        aspect_score = 1.0
    elif aspect <= 6.0:
        aspect_score = float(1.0 - (aspect - 3.0) * 0.15)
    else:
        aspect_score = max(0.1, float(1.0 - (aspect - 3.0) * 0.10))

    # Weighted sub-score
    sub = (
        0.35 * compactness_score +
        0.25 * convexity_score +
        0.20 * area_score +
        0.20 * aspect_score
    )

    components = {
        "compactness_score": round(compactness_score, 4),
        "convexity_score": round(convexity_score, 4),
        "area_plausibility_score": round(area_score, 4),
        "aspect_ratio_score": round(aspect_score, 4),
    }

    return round(float(np.clip(sub, 0.0, 1.0)), 4), components


def _building_subscore(evidence: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Building evidence sub-score from:
      - Building containment ratio (target: 1.0)
      - Coverage ratio plausibility (target: 0.15-0.65)
      - Building area relative to parcel
    """
    containment = evidence.get("containment_ratio", 0.0)
    coverage = evidence.get("coverage_ratio", 0.0)
    coverage_ok = evidence.get("coverage_plausible", False)
    fully_contained = evidence.get("building_fully_contained", False)

    # Containment score: direct mapping
    containment_score = float(containment)

    # Coverage score: bell-curve preference for 0.15-0.65 range
    if coverage_ok:
        # Within plausible range
        if 0.20 <= coverage <= 0.55:
            cov_score = 1.0
        else:
            cov_score = 0.8
    else:
        if coverage < 0.05:
            cov_score = 0.2  # Too sparse
        elif coverage > 0.80:
            cov_score = 0.3  # Oversaturated
        else:
            cov_score = 0.5

    # Bonus for full containment
    contain_bonus = 0.1 if fully_contained else 0.0

    sub = (
        0.50 * containment_score +
        0.40 * cov_score +
        contain_bonus
    )

    components = {
        "containment_score": round(containment_score, 4),
        "coverage_plausibility_score": round(cov_score, 4),
        "full_containment_bonus": round(contain_bonus, 4),
    }

    return round(float(np.clip(sub, 0.0, 1.0)), 4), components


def _road_access_subscore(evidence: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Road access sub-score from:
      - Road frontage length (target: >= 4.0 m)
      - Road distance (target: 0.0 m direct contact)
      - Road evidence score from Phase 13B/D
      - Source evidence quality (classical, DL, or both)
    """
    road_dist = evidence.get("road_distance_m", None)
    frontage = evidence.get("road_frontage_m", 0.0)
    rd_score_raw = evidence.get("road_evidence_score", None)
    touches = evidence.get("road_touches", False)
    source = evidence.get("nearest_road_source", "unknown")

    # If no road data available, assign neutral score
    if road_dist is None and rd_score_raw is None:
        return 0.50, {
            "frontage_score": 0.5,
            "distance_score": 0.5,
            "source_quality_score": 0.5,
            "note": "no_road_data_available",
        }

    # Frontage score
    if frontage >= 8.0:
        frontage_score = 1.0
    elif frontage >= 4.0:
        frontage_score = 0.85
    elif frontage > 0:
        frontage_score = 0.5
    else:
        frontage_score = 0.15

    # Distance score
    if touches or (road_dist is not None and road_dist < 0.2):
        distance_score = 1.0
    elif road_dist is not None and road_dist <= 5.0:
        distance_score = 0.7
    elif road_dist is not None and road_dist <= 15.0:
        distance_score = 0.4
    else:
        distance_score = 0.15

    # Source evidence quality
    if source == "both":
        source_score = 1.0
    elif source == "classical":
        source_score = 0.85
    elif source == "deep_learning":
        source_score = 0.70
    else:
        source_score = 0.50

    sub = (
        0.40 * frontage_score +
        0.35 * distance_score +
        0.25 * source_score
    )

    components = {
        "frontage_score": round(frontage_score, 4),
        "distance_score": round(distance_score, 4),
        "source_quality_score": round(source_score, 4),
    }

    return round(float(np.clip(sub, 0.0, 1.0)), 4), components


def _topology_subscore(evidence: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Topological plausibility sub-score from:
      - Validity status (valid / repaired / flagged)
      - Whether geometry was repaired (minor penalty)
    """
    is_valid = evidence.get("is_valid", False)
    was_repaired = evidence.get("was_repaired", False)
    is_flagged = evidence.get("is_flagged", False)

    if is_valid and not was_repaired:
        topo_score = 1.0
    elif is_valid and was_repaired:
        topo_score = 0.80
    elif is_flagged:
        topo_score = 0.20
    else:
        topo_score = 0.50

    components = {
        "validity_score": round(topo_score, 4),
        "was_repaired": was_repaired,
        "is_flagged": is_flagged,
    }

    return round(topo_score, 4), components


class ParcelConfidenceEngine:
    """
    Deterministic confidence scoring engine for preliminary parcel candidates.

    Produces for each candidate:
      - confidence: float [0, 1]
      - confidence_band: str (HIGH, MODERATE, LOW, VERY_LOW)
      - confidence_components: dict of sub-scores with full explainability
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or WEIGHTS.copy()

        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            for k in self.weights:
                self.weights[k] /= total

    def score_single(
        self,
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Computes confidence for a single parcel candidate.

        Args:
            evidence: Evidence record from EvidenceAssembler.

        Returns:
            Dictionary with confidence, confidence_band, and confidence_components.
        """
        geom_score, geom_components = _geometry_subscore(evidence.get("geometry", {}))
        bldg_score, bldg_components = _building_subscore(evidence.get("building", {}))
        road_score, road_components = _road_access_subscore(evidence.get("road_access", {}))
        topo_score, topo_components = _topology_subscore(evidence.get("topology", {}))

        # Weighted composite
        confidence = (
            self.weights["geometry"]    * geom_score +
            self.weights["building"]    * bldg_score +
            self.weights["road_access"] * road_score +
            self.weights["topology"]    * topo_score
        )
        confidence = round(float(np.clip(confidence, 0.0, 1.0)), 4)
        band = _classify_band(confidence)

        return {
            "confidence": confidence,
            "confidence_band": band,
            "confidence_components": {
                "geometry_quality": {
                    "score": geom_score,
                    "weight": self.weights["geometry"],
                    "components": geom_components,
                },
                "building_evidence": {
                    "score": bldg_score,
                    "weight": self.weights["building"],
                    "components": bldg_components,
                },
                "road_access": {
                    "score": road_score,
                    "weight": self.weights["road_access"],
                    "components": road_components,
                },
                "topological_plausibility": {
                    "score": topo_score,
                    "weight": self.weights["topology"],
                    "components": topo_components,
                },
            },
        }

    def score_batch(
        self,
        evidence_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Scores all parcel candidates in batch.

        Returns:
            List of confidence result dictionaries.
        """
        return [self.score_single(ev) for ev in evidence_list]

    def compute_distribution_summary(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Computes population-level confidence distribution statistics.
        """
        scores = [r["confidence"] for r in results]
        bands = [r["confidence_band"] for r in results]

        n = len(scores)
        if n == 0:
            return {"total_candidates": 0}

        arr = np.array(scores)
        band_counts = {}
        for b in ["HIGH", "MODERATE", "LOW", "VERY_LOW"]:
            band_counts[b] = int(bands.count(b))

        return {
            "total_candidates": n,
            "mean_confidence": round(float(arr.mean()), 4),
            "median_confidence": round(float(np.median(arr)), 4),
            "std_confidence": round(float(arr.std()), 4),
            "min_confidence": round(float(arr.min()), 4),
            "max_confidence": round(float(arr.max()), 4),
            "percentile_25": round(float(np.percentile(arr, 25)), 4),
            "percentile_75": round(float(np.percentile(arr, 75)), 4),
            "band_distribution": band_counts,
            "band_percentages": {
                k: round(100.0 * v / n, 2) for k, v in band_counts.items()
            },
        }
