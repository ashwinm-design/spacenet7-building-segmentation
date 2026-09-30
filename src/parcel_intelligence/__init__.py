"""
src/parcel_intelligence/__init__.py

Phase 14A — Parcel Intelligence & Explainable Confidence Audit
SIH26012

Deterministic, evidence-based confidence scoring and uncertainty flagging
for AI-generated preliminary urban parcel candidates.

IMPORTANT DISCLAIMERS:
- Confidence scores measure AI inference quality, NOT legal cadastral certainty.
- All flags indicate areas requiring human professional review.
- No ground truth cadastral data exists; all metrics are computed from
  measurable AI-inferred evidence only.
"""

from .confidence import ParcelConfidenceEngine
from .evidence import EvidenceAssembler
from .flags import UncertaintyFlagger
from .explainability import ExplainabilityReport

__all__ = [
    "ParcelConfidenceEngine",
    "EvidenceAssembler",
    "UncertaintyFlagger",
    "ExplainabilityReport",
]
