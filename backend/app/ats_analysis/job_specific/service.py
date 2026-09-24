"""Job-Specific ATS Coverage analysis service.

Transient, deterministic, explainable measurement of how completely the job
description's terminology is represented in a parsed resume. No persistence,
no filesystem access, no database, no embeddings, no external network, no LLM,
no logging of resume or job content.

The returned score is a heuristic representation signal — see the disclaimer
in ``AnalysisMetadata``.
"""

from __future__ import annotations

from app.ats_analysis.job_specific.analyzers import analyze_job_specific
from app.ats_analysis.job_specific.rules import (
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    DISCLAIMER,
    METHOD_NAME,
    VERSION,
    WEIGHTS,
)
from app.ats_analysis.job_specific.schemas import (
    JobSpecificATSResult,
    TermMatch,
)
from app.ats_analysis.job_specific.scorer import (
    compute_overall,
    score_label,
    score_label_docs,
)
from app.ats_analysis.schemas import (
    AnalysisMetadata,
    CategoryScore,
)
from app.job_parsing.schemas import JobDescription
from app.parsing.schemas import Resume


def analyze_job_specific_ats(
    resume: Resume, job: JobDescription
) -> JobSpecificATSResult:
    """Run the full deterministic coverage analysis for a resume + job pair.

    Both inputs are treated as immutable and are never modified or stored.
    """
    scores, coverage, term_matches, findings = analyze_job_specific(resume, job)

    overall, applied_weights = compute_overall(scores)

    category_scores = [
        CategoryScore(
            key=category,
            label=CATEGORY_LABELS[category],
            score=scores.get(category),
            applicable=scores.get(category) is not None,
            weight=applied_weights.get(category, 0.0),
            max_weight=WEIGHTS[category],
        )
        for category in CATEGORY_ORDER
    ]

    ordered_matches = _order_term_matches(term_matches)

    metadata = AnalysisMetadata(
        method=METHOD_NAME,
        version=VERSION,
        weights=dict(WEIGHTS),
        applied_weights=applied_weights,
        score_labels=score_label_docs(),
        disclaimer=DISCLAIMER,
    )

    return JobSpecificATSResult(
        overall_score=overall,
        score_label=score_label(overall),
        category_scores=category_scores,
        coverage=coverage,
        term_matches=ordered_matches,
        findings=findings,
        metadata=metadata,
    )


def _order_term_matches(matches: list[TermMatch]) -> list[TermMatch]:
    """Deterministic ordering: required < preferred < phrase, then present
    before absent, then original (stable) order."""
    origin_rank = {"required": 0, "preferred": 1, "phrase": 2}
    return sorted(
        matches,
        key=lambda m: (
            origin_rank[m.origin.value],
            m.match_type.value == "absent",
        ),
    )
