"""Deterministic overall scoring for ATS readiness.

The ATS Readiness Score is a weighted average of the applicable category
scores. When a category is not applicable (e.g. there is no work experience),
its weight is redistributed proportionally across the applicable categories —
the resume is never penalised for the absence of a section type it does not
need. `overall_score` is `None` only if no category is applicable (extremely
unlikely, since contact/structure/parsing always apply).

All arithmetic is deterministic. The result is a heuristic quality signal, not
a probability of passing any specific ATS or of being hired.
"""

from __future__ import annotations

from app.ats_analysis.rules import CATEGORY_ORDER, SCORE_LABELS_DOC, WEIGHTS

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
        weight = WEIGHTS[category]
        applied[category] = weight
        weighted_sum += weight * score

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
