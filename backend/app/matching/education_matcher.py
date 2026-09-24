"""Deterministic education matching at the level of recognised degrees.

Degree strings are classified into a hierarchy (doctorate > master > bachelor >
diploma) using explicit, documented patterns. Equivalences such as
Bachelor's ↔ B.Tech ↔ B.E. and Master's ↔ M.Tech ↔ MCA are recognised through
this classification. No unrelated degrees are treated as equivalent.

When the job states no education requirement the criterion is marked as not
required instead of penalising the candidate.
"""

from __future__ import annotations

import re

from app.job_parsing.schemas import JobDescription
from app.matching.schemas import EducationMatch
from app.parsing.schemas import Resume

_DEGREE_LEVELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "doctorate",
        (
            r"\bph\.?d\.?\b",
            r"\bdoctor(?:al|ate)?\b",
            r"\bd\.?sc\b",
        ),
    ),
    (
        "master",
        (
            r"\bm\.?a\.?\b",
            r"\bm\.?s\.?\b",
            r"\bms\b",
            r"\bm\.?tech\.?\b",
            r"\bmtech\b",
            r"\bm\.?e\.?\b",
            r"\bmca\b",
            r"\bm\.?sc\.?\b",
            r"\bm\.?ba\.?\b",
            r"\bm\.?com\.?\b",
            r"\bmaster(?:'s)?\b",
        ),
    ),
    (
        "bachelor",
        (
            r"\bb\.?a\.?\b",
            r"\bb\.?s\.?\b",
            r"\bb\.?sc\.?\b",
            r"\bb\.?e\.?\b",
            r"\bb\.?tech\.?\b",
            r"\bbtech\b",
            r"\bb\.?sc\.?\b",
            r"\bbca\b",
            r"\bbba\b",
            r"\bb\.?com\.?\b",
            r"\bbachelor(?:'s)?\b",
        ),
    ),
    (
        "diploma",
        (
            r"\bdiploma\b",
            r"\bpolytechnic\b",
            r"\bassociate(?:'s)?\b",
        ),
    ),
)

_LEVEL_ORDER: dict[str, int] = {
    "diploma": 1,
    "bachelor": 2,
    "master": 3,
    "doctorate": 4,
}


def classify_degree_level(degree: str | None) -> str | None:
    """Return the highest recognised level in a degree string, or None."""
    if not degree:
        return None
    text = degree.strip().lower()
    for level, patterns in _DEGREE_LEVELS:
        for pattern in patterns:
            if re.search(pattern, text):
                return level
    return None


def _highest_level(levels: list[str | None]) -> str | None:
    present = [lvl for lvl in levels if lvl is not None]
    if not present:
        return None
    return max(present, key=lambda lvl: _LEVEL_ORDER[lvl])


def candidate_highest_level(resume: Resume) -> str | None:
    return _highest_level(
        [classify_degree_level(entry.degree) for entry in resume.education]
    )


def required_education_level(job: JobDescription) -> str | None:
    return _highest_level(
        [classify_degree_level(statement) for statement in job.education_requirements]
    )


def match_education(resume: Resume, job: JobDescription) -> EducationMatch:
    required_level = required_education_level(job)
    candidate_level = candidate_highest_level(resume)

    if required_level is None:
        return EducationMatch(
            score=None,
            required_level=None,
            candidate_level=candidate_level,
            matched=None,
            assessment=(
                "Job states no classifiable education requirement — not evaluated."
            ),
        )

    if candidate_level is None:
        return EducationMatch(
            score=0.0,
            required_level=required_level,
            candidate_level=None,
            matched=False,
            assessment=(
                f"Job requires {required_level}-level education; resume lists no "
                "recognisable degrees."
            ),
        )

    matched = _LEVEL_ORDER[candidate_level] >= _LEVEL_ORDER[required_level]
    return EducationMatch(
        score=100.0 if matched else 0.0,
        required_level=required_level,
        candidate_level=candidate_level,
        matched=matched,
        assessment=(
            f"{required_level.capitalize()}-level requirement "
            + ("satisfied" if matched else "not satisfied")
            + f" (resume shows {candidate_level}-level education)."
        ),
    )
