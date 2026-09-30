"""
src/parcel_intelligence/explainability.py

Phase 14A — Explainability Report Generator for SIH26012

Generates human-readable, structured audit reports that explain
exactly WHY each parcel candidate received its confidence score and flags.

Outputs:
  - Per-candidate natural-language explanation paragraphs
  - Top-N and Bottom-N case identification
  - Actionable review recommendations
  - Population-level audit narrative

SCIENTIFIC INTEGRITY:
  - All explanations are deterministic and reproducible.
  - No ground truth data is fabricated.
  - Language explicitly states "AI-generated preliminary candidate".
"""

from typing import Dict, Any, List, Optional
import numpy as np


class ExplainabilityReport:
    """
    Generates structured explainability artifacts for the parcel intelligence audit.
    """

    def __init__(self):
        pass

    def explain_single(
        self,
        parcel_id: int,
        evidence: Dict[str, Any],
        confidence_result: Dict[str, Any],
        flags: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Produces a structured explanation for a single parcel candidate.

        Returns:
            Dictionary with summary, detailed_explanation, strengths, weaknesses, recommendation.
        """
        conf = confidence_result["confidence"]
        band = confidence_result["confidence_band"]
        components = confidence_result.get("confidence_components", {})

        # Identify strengths (sub-scores >= 0.75) and weaknesses (sub-scores < 0.50)
        strengths = []
        weaknesses = []
        for dim_name, dim_data in components.items():
            score = dim_data.get("score", 0.0)
            weight = dim_data.get("weight", 0.0)
            if score >= 0.75:
                strengths.append({
                    "dimension": dim_name,
                    "score": score,
                    "weight": weight,
                    "label": self._dimension_label(dim_name),
                })
            elif score < 0.50:
                weaknesses.append({
                    "dimension": dim_name,
                    "score": score,
                    "weight": weight,
                    "label": self._dimension_label(dim_name),
                })

        # Generate natural-language summary
        summary = self._generate_summary(parcel_id, conf, band, strengths, weaknesses, flags)

        # Generate recommendation
        recommendation = self._generate_recommendation(band, flags, weaknesses)

        return {
            "parcel_id": parcel_id,
            "confidence": conf,
            "confidence_band": band,
            "summary": summary,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "flags": [f["flag_id"] for f in flags],
            "flag_count": len(flags),
            "recommendation": recommendation,
        }

    def identify_top_bottom(
        self,
        explanations: List[Dict[str, Any]],
        n: int = 10,
    ) -> Dict[str, Any]:
        """
        Identifies top-N (highest confidence) and bottom-N (lowest confidence) candidates.
        """
        sorted_by_conf = sorted(explanations, key=lambda x: x["confidence"], reverse=True)

        top_n = sorted_by_conf[:n]
        bottom_n = sorted_by_conf[-n:] if len(sorted_by_conf) >= n else sorted_by_conf

        return {
            "top_n": [{
                "parcel_id": e["parcel_id"],
                "confidence": e["confidence"],
                "confidence_band": e["confidence_band"],
                "strengths": [s["dimension"] for s in e["strengths"]],
                "flag_count": e["flag_count"],
            } for e in top_n],
            "bottom_n": [{
                "parcel_id": e["parcel_id"],
                "confidence": e["confidence"],
                "confidence_band": e["confidence_band"],
                "weaknesses": [w["dimension"] for w in e["weaknesses"]],
                "flags": e["flags"],
                "flag_count": e["flag_count"],
            } for e in bottom_n],
        }

    def generate_population_narrative(
        self,
        distribution_summary: Dict[str, Any],
        flag_summary: Dict[str, Any],
        top_bottom: Dict[str, Any],
    ) -> str:
        """
        Generates a structured narrative summarizing the entire parcel audit.
        """
        n = distribution_summary.get("total_candidates", 0)
        mean_conf = distribution_summary.get("mean_confidence", 0.0)
        median_conf = distribution_summary.get("median_confidence", 0.0)
        std_conf = distribution_summary.get("std_confidence", 0.0)
        bands = distribution_summary.get("band_distribution", {})
        band_pcts = distribution_summary.get("band_percentages", {})

        flagged_pct = flag_summary.get("flag_rate_pct", 0.0)
        total_flags = flag_summary.get("total_flags_raised", 0)
        flag_dist = flag_summary.get("flag_type_distribution", {})

        lines = [
            "# PHASE 14A — PARCEL INTELLIGENCE AUDIT NARRATIVE",
            "",
            "## Population Summary",
            "",
            f"The Phase 14A audit evaluated **{n} AI-generated preliminary parcel candidates**.",
            f"These candidates were produced by the SIH26012 hybrid pipeline (Phase 13D) using",
            f"boundary-aware U-Net building segmentation, hybrid classical+DL road extraction,",
            f"and Voronoi-based spatial partitioning.",
            "",
            "### Confidence Distribution",
            "",
            f"| Statistic | Value |",
            f"|-----------|-------|",
            f"| Mean Confidence | {mean_conf:.4f} |",
            f"| Median Confidence | {median_conf:.4f} |",
            f"| Std. Deviation | {std_conf:.4f} |",
            f"| Min Confidence | {distribution_summary.get('min_confidence', 0):.4f} |",
            f"| Max Confidence | {distribution_summary.get('max_confidence', 0):.4f} |",
            f"| P25 | {distribution_summary.get('percentile_25', 0):.4f} |",
            f"| P75 | {distribution_summary.get('percentile_75', 0):.4f} |",
            "",
            "### Band Distribution",
            "",
            f"| Band | Count | Percentage |",
            f"|------|-------|------------|",
        ]

        for band_name in ["HIGH", "MODERATE", "LOW", "VERY_LOW"]:
            cnt = bands.get(band_name, 0)
            pct = band_pcts.get(band_name, 0.0)
            lines.append(f"| {band_name} | {cnt} | {pct:.1f}% |")

        lines.extend([
            "",
            "### Uncertainty Flags",
            "",
            f"- **{flagged_pct:.1f}%** of candidates have at least one uncertainty flag.",
            f"- **{total_flags}** total flags raised across all candidates.",
            "",
            f"| Flag Type | Count |",
            f"|-----------|-------|",
        ])

        for flag_name, cnt in sorted(flag_dist.items(), key=lambda x: -x[1]):
            lines.append(f"| {flag_name} | {cnt} |")

        lines.extend([
            "",
            "### Top Performing Candidates",
            "",
        ])
        for item in top_bottom.get("top_n", [])[:5]:
            pid = item["parcel_id"]
            c = item["confidence"]
            s = ", ".join(item.get("strengths", []))
            lines.append(f"- Parcel {pid}: confidence={c:.4f}, strengths=[{s}]")

        lines.extend([
            "",
            "### Candidates Requiring Review",
            "",
        ])
        for item in top_bottom.get("bottom_n", [])[:5]:
            pid = item["parcel_id"]
            c = item["confidence"]
            f = ", ".join(item.get("flags", []))
            lines.append(f"- Parcel {pid}: confidence={c:.4f}, flags=[{f}]")

        lines.extend([
            "",
            "---",
            "",
            "**DISCLAIMER:** All confidence scores measure AI inference quality and geometric",
            "plausibility. They do NOT represent legal cadastral certainty or property ownership.",
            "All boundaries are preliminary AI-assisted candidates requiring professional survey validation.",
        ])

        return "\n".join(lines)

    def _generate_summary(
        self,
        parcel_id: int,
        conf: float,
        band: str,
        strengths: List[Dict[str, Any]],
        weaknesses: List[Dict[str, Any]],
        flags: List[Dict[str, Any]],
    ) -> str:
        """Generates natural-language summary for a single candidate."""
        parts = [
            f"AI-generated preliminary parcel candidate #{parcel_id} "
            f"achieved a confidence score of {conf:.4f} ({band})."
        ]

        if strengths:
            s_labels = [s["label"] for s in strengths]
            parts.append(f"Strong evidence in: {', '.join(s_labels)}.")

        if weaknesses:
            w_labels = [w["label"] for w in weaknesses]
            parts.append(f"Weak evidence in: {', '.join(w_labels)}.")

        if flags:
            flag_ids = [f["flag_id"] for f in flags]
            parts.append(f"Flagged for review: {', '.join(flag_ids)}.")

        return " ".join(parts)

    def _generate_recommendation(
        self,
        band: str,
        flags: List[Dict[str, Any]],
        weaknesses: List[Dict[str, Any]],
    ) -> str:
        """Generates actionable recommendation based on band and flags."""
        critical_flags = [f for f in flags if f.get("severity") == "CRITICAL"]

        if critical_flags:
            return "CRITICAL: Requires immediate human inspection due to topological or geometric issues."

        if band == "HIGH":
            if not flags:
                return "ACCEPT: High-confidence candidate with no uncertainty flags. Suitable for preliminary map inclusion."
            else:
                return "REVIEW: High confidence but has minor flags. Quick review recommended."

        if band == "MODERATE":
            if len(flags) <= 1:
                return "REVIEW: Moderate confidence. Targeted review of flagged dimensions recommended."
            else:
                return "REVIEW: Moderate confidence with multiple flags. Detailed professional review recommended."

        if band == "LOW":
            return "CAUTION: Low confidence. Multiple evidence dimensions are weak. Manual boundary assessment recommended."

        return "REJECT-REVIEW: Very low confidence. Candidate should not be used without comprehensive manual verification."

    def _dimension_label(self, dim_name: str) -> str:
        """Maps dimension keys to human-readable labels."""
        labels = {
            "geometry_quality": "Geometry Quality",
            "building_evidence": "Building Evidence",
            "road_access": "Road Access",
            "topological_plausibility": "Topological Plausibility",
        }
        return labels.get(dim_name, dim_name)
