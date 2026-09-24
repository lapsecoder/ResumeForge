"""Deterministic overall scoring for Job-Specific ATS Coverage.

The score is a weighted average of the applicable category scores. When a
category is not applicable (e.g. the job lists no preferred skills, or the
resume matched no terms so evidence representation cannot be assessed), its
weight is redistributed proportionally across the applicable categories — the
candidate is never penalised for something the pair does not need. The score
is ``None`` only if no category is applicable (job and resume contain no
extractable terminology).

All arithmetic is deterministic. The result measures terminology
representation; it is not a probability of passing any ATS or of being hired.
"""

from __future__ import annotations

from app.ats_analysis.job_specific.rules import (
    CATEGORY_ORDER,
    SCORE_LABELS_DOC,
    WEIGHTS,
)

_SCORE_LABEL_BOUNDS: list[tuple[str, float, float]] = [
    ("Weak", 0.0, 50.0),
    ("Needs Improvement", 50.0, 65.0),
    ("Good", 65.0, 80.0),
    ("Strong", 80.0, 100.0),
]


def compute_overall(
    scores: dict[str, float | None],
) -> tuple[float | None, dict[str, float]]:
    """Return (overall_score, applied weights after redistribution)."""
    applied: dict[str, float] = {}
    weighted_sum = 0.0
    for category in CATEGORY_ORDER:
        score = scores.get(category)
        if score is None:
            continue
        applied[category] = WEIGHTS[category]
        weighted_sum += WEIGHTS[category] * score

    if not applied:
        return None, {}

    total = sum(applied.values())
    return round(weighted_sum / total, 1), applied


def score_label(overall: float | None) -> str | None:
    """Map an overall score to its documented label."""
    if overall is None:
        return None
    for label, lower, upper in _SCORE_LABEL_BOUNDS:
        if lower <= overall < upper:
            return label
    return "Strong"


def score_label_docs() -> dict[str, str]:
    """Return a copy of the human-readable threshold documentation."""
    return dict(SCORE_LABELS_DOC)
