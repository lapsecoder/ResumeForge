"""ATS Readiness / resume-quality analysis service.

Transient, deterministic, explainable analysis of a parsed resume WITHOUT a
job description. No persistence, no filesystem access, no database, no
embeddings, no external network, no LLM, no logging of resume content.

The returned score is a heuristic quality signal — see the disclaimer in
``AnalysisMetadata``.
"""

from __future__ import annotations

from typing import Callable

from app.ats_analysis.analyzers import (
    analyze_bullets,
    analyze_contact,
    analyze_dates,
    analyze_education,
    analyze_experience,
    analyze_parsing,
    analyze_projects_certs,
    analyze_quantified,
    analyze_skills,
    analyze_structure,
)
from app.ats_analysis.rules import (
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    DISCLAIMER,
    METHOD_NAME,
    VERSION,
    WEIGHTS,
)
from app.ats_analysis.schemas import (
    AnalysisMetadata,
    ATSReadinessResult,
    CategoryScore,
    Finding,
)
from app.ats_analysis.scorer import compute_overall, score_label, score_label_docs
from app.parsing.schemas import Resume

_ANALYZERS: dict[str, Callable[[Resume], tuple[float | None, list[Finding]]]] = {
    "contact": analyze_contact,
    "structure": analyze_structure,
    "experience": analyze_experience,
    "bullets": analyze_bullets,
    "quantified": analyze_quantified,
    "skills": analyze_skills,
    "education": analyze_education,
    "projects_certs": analyze_projects_certs,
    "dates": analyze_dates,
    "parsing": analyze_parsing,
}


def analyze_resume(resume: Resume) -> ATSReadinessResult:
    """Run the full deterministic ATS readiness analysis on a resume.

    The resume is treated as immutable and is never modified or stored.
    """
    results: dict[str, float | None] = {}
    findings: list[Finding] = []

    for category in CATEGORY_ORDER:
        score, category_findings = _ANALYZERS[category](resume)
        results[category] = score
        findings.extend(category_findings)

    overall, applied_weights = compute_overall(results)

    category_scores = [
        CategoryScore(
            key=category,
            label=CATEGORY_LABELS[category],
            score=results[category],
            applicable=results[category] is not None,
            weight=applied_weights.get(category, 0.0),
            max_weight=WEIGHTS[category],
        )
        for category in CATEGORY_ORDER
    ]

    metadata = AnalysisMetadata(
        method=METHOD_NAME,
        version=VERSION,
        weights=dict(WEIGHTS),
        applied_weights=applied_weights,
        score_labels=score_label_docs(),
        disclaimer=DISCLAIMER,
    )

    return ATSReadinessResult(
        overall_score=overall,
        score_label=score_label(overall),
        category_scores=category_scores,
        findings=findings,
        metadata=metadata,
    )
