"""Hybrid matching service (Phase 5C).

Orchestrates the authoritative deterministic baseline (5A) and the supporting
local semantic layer (5B) into a transient HybridMatchResult. The semantic
model is never a single point of failure: if it is missing, fails to load, or
fails inference, the endpoint still returns a deterministic-backed result with
the semantic availability clearly marked. Nothing is persisted, logged, or
sent externally.
"""

from __future__ import annotations

import datetime as _dt
import logging

from app.hybrid_matching.insights import build_semantic_insights
from app.hybrid_matching.schemas import (
    ComponentScores,
    HybridMatchResult,
    HybridMetadata,
    SemanticAvailability,
)
from app.hybrid_matching.scoring import HYBRID_WEIGHTS, evidence_quality, hybrid_overall
from app.job_parsing.schemas import JobDescription
from app.matching.schemas import MatchResult
from app.matching.service import match_resume_to_job
from app.parsing.schemas import Resume
from app.semantic_matching.model import EmbeddingProvider, ModelLoadError, SemanticError
from app.semantic_matching.schemas import SemanticMatchResult
from app.semantic_matching.service import compute_semantic_match

logger = logging.getLogger(__name__)

SEMANTIC_NOTE_UNKNOWN = (
    "Semantic evidence could not be produced; the hybrid result is supported "
    "by the deterministic signal only."
)


def compute_hybrid_match(
    resume: Resume,
    job: JobDescription,
    *,
    reference_date: _dt.date | None = None,
    provider: EmbeddingProvider | None = None,
) -> HybridMatchResult:
    """Return a transient, explainable HybridMatchResult for resume vs job.

    ``reference_date`` (default today) keeps all date-based calculations
    traceable and deterministic in tests. ``provider`` lets tests inject a
    fake embedding provider; when omitted, the semantic service uses its
    process-wide lazy provider.
    """
    reference = reference_date or _dt.date.today()

    deterministic = match_resume_to_job(resume, job, reference_date=reference)
    semantic, availability = _attempt_semantic(resume, job, provider)

    semantic_overall = semantic.overall_similarity if semantic else None
    overall, mode = hybrid_overall(deterministic.overall_score, semantic_overall)

    quality = evidence_quality(
        mode,
        len(deterministic.metadata.applied_weights),
        semantic_overall is not None,
    )

    insights = build_semantic_insights(deterministic, semantic, job)

    return HybridMatchResult(
        overall_score=overall,
        deterministic=deterministic,
        semantic=semantic,
        component_scores=_component_scores(deterministic, semantic, overall),
        matched_requirements=list(deterministic.matched_requirements),
        missing_required=list(deterministic.skill_match.missing_required),
        strengths=list(deterministic.strengths),
        gaps=list(deterministic.gaps),
        semantic_insights=insights,
        metadata=HybridMetadata(
            mode=mode,
            weights=dict(HYBRID_WEIGHTS),
            evidence_quality=quality,
            semantic_availability=availability,
            semantic_metadata=semantic.metadata if semantic else None,
            reference_date=reference,
        ),
    )


def _component_scores(
    deterministic: MatchResult,
    semantic: SemanticMatchResult | None,
    hybrid_overall_score: float | None,
) -> ComponentScores:
    """Map the native signals into the documented ComponentScores layout.

    Every component reports what was actually evaluable; unavailable values
    stay ``None`` instead of being invented as zeros.
    """
    return ComponentScores(
        deterministic_overall=deterministic.overall_score,
        semantic_overall=semantic.overall_similarity if semantic else None,
        hybrid_overall=hybrid_overall_score,
        deterministic_skills=deterministic.skill_match.score,
        semantic_skills=semantic.skill_similarity if semantic else None,
        deterministic_experience=deterministic.experience_match.score,
        semantic_experience=semantic.experience_similarity if semantic else None,
        deterministic_education=deterministic.education_match.score,
        semantic_qualification=(
            semantic.qualification_similarity if semantic else None
        ),
    )


def _attempt_semantic(
    resume: Resume,
    job: JobDescription,
    provider: EmbeddingProvider | None,
) -> tuple[SemanticMatchResult | None, SemanticAvailability]:
    """Run the semantic layer without letting it break deterministic matching.

    A missing model, an inference failure, or any unexpected error degrades to
    a deterministic-only result; the failure mode is surfaced in the
    availability descriptor. Error paths never log user content.
    """
    try:
        result = compute_semantic_match(resume, job, provider=provider)
    except ModelLoadError:
        logger.warning(
            "Hybrid: semantic model unavailable; continuing deterministic only."
        )
        return None, SemanticAvailability(
            available=False,
            status="model_unavailable",
            note=(
                "The local semantic model is unavailable; the hybrid result "
                "uses the deterministic signal only."
            ),
        )
    except SemanticError:
        logger.warning(
            "Hybrid: semantic inference failed; continuing deterministic only."
        )
        return None, SemanticAvailability(
            available=False,
            status="inference_failed",
            note=(
                "Local semantic inference failed; the hybrid result uses the "
                "deterministic signal only."
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive degradation
        logger.warning(
            "Hybrid: unexpected semantic failure (%s); continuing deterministic "
            "only.",
            type(exc).__name__,
        )
        return None, SemanticAvailability(
            available=False,
            status="error",
            note=SEMANTIC_NOTE_UNKNOWN,
        )

    if result.overall_similarity is None:
        return result, SemanticAvailability(
            available=True,
            status="no_content",
            note=result.note,
        )
    return result, SemanticAvailability(
        available=True,
        status="available",
        note="",
    )
