"""Hybrid scoring rules (Phase 5C).

The hybrid formula blends the authoritative deterministic signal with the
supporting semantic signal:

    deterministic_score = Phase 5A overall score          (0-100)
    semantic_score      = Phase 5B overall similarity     (0-1)  -> *100
    hybrid_score        = 0.70 * deterministic + 0.30 * semantic

Blending happens ONLY when both signals are meaningful. Otherwise:

    - deterministic only      -> hybrid equals the deterministic score.
    - semantic only           -> hybrid is None: semantic relatedness alone
                                 is not treated as a verified match score.
    - neither                 -> hybrid is None.

Weights document the principle that deterministic evidence stays
authoritative: the semantic signal can nudge, but never dominate or inflate,
explicit requirement satisfaction.

The Hybrid Match Score is a heuristic relevance score, not a hiring
probability, an ATS probability, or an employment prediction.
"""

from __future__ import annotations

from app.hybrid_matching.schemas import EvidenceQuality, HybridMatchMode

HYBRID_DETERMINISTIC_WEIGHT = 0.70
HYBRID_SEMANTIC_WEIGHT = 0.30

#: Documented, immutable weights used by the blend.
HYBRID_WEIGHTS: dict[str, float] = {
    "deterministic": HYBRID_DETERMINISTIC_WEIGHT,
    "semantic": HYBRID_SEMANTIC_WEIGHT,
}


def semantic_on_100(semantic_overall: float) -> float:
    """Map semantic similarity [0, 1] onto the 0-100 scale used by the blend."""
    return semantic_overall * 100.0


def hybrid_overall(
    deterministic_overall: float | None,
    semantic_overall: float | None,
) -> tuple[float | None, HybridMatchMode]:
    """Return (hybrid score, mode) per the documented formula.

    ``hybrid`` results round to one decimal place; ``deterministic-only``
    returns the deterministic value untouched so 5A semantics are preserved.
    """
    if deterministic_overall is not None and semantic_overall is not None:
        blended = (
            HYBRID_DETERMINISTIC_WEIGHT * deterministic_overall
            + HYBRID_SEMANTIC_WEIGHT * semantic_on_100(semantic_overall)
        )
        return round(blended, 1), "hybrid"
    if deterministic_overall is not None:
        return deterministic_overall, "deterministic-only"
    if semantic_overall is not None:
        return None, "semantic-only"
    return None, "no-evidence"


def evidence_quality(
    mode: HybridMatchMode,
    evaluable_deterministic_components: int,
    semantic_present: bool,
) -> EvidenceQuality:
    """Rate how much structured information supported the result.

    This is an evidence-quality signal, NOT a statistical confidence:
    - high    -> both signals present with at least two deterministic criteria.
    - medium  -> at least one meaningful signal.
    - limited -> no meaningful evidence at all.
    """
    if (
        mode == "hybrid"
        and evaluable_deterministic_components >= 2
        and semantic_present
    ):
        return "high"
    if mode == "no-evidence":
        return "limited"
    return "medium"
