"""Hybrid matching (Phase 5C): compose the authoritative deterministic
baseline (5A) with the supporting local semantic layer (5B) into a single
transient, explainable HybridMatchResult.

The deterministic layer stays authoritative for explicit requirements;
semantic similarity only supplements context-level matching and never implies
skill possession. Nothing here is persisted, logged, or sent externally.
"""

from app.hybrid_matching.schemas import (
    ComponentScores,
    EvidenceQuality,
    HybridMatchMode,
    HybridMatchResult,
    HybridMetadata,
    SemanticAvailability,
    SemanticInsight,
)
from app.hybrid_matching.service import compute_hybrid_match

__all__ = [
    "ComponentScores",
    "EvidenceQuality",
    "HybridMatchMode",
    "HybridMetadata",
    "HybridMatchResult",
    "SemanticAvailability",
    "SemanticInsight",
    "compute_hybrid_match",
]
