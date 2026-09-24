"""Deterministic structured-attribute helpers.

These small pure functions shape resume/job attributes into the input columns
the structured feature blocks consume. All of them are deterministic and
stateless so feature versions only change when the rules here change
(``feature_version`` in the experiment config).
"""

from __future__ import annotations

import re

from ml.preprocessing.text import normalize


def seniority_tier(years_experience: float) -> str:
    """Bucket years of experience into a coarse seniority tier."""
    if years_experience < 0:
        raise ValueError("years_experience must be >= 0")
    if years_experience < 3:
        return "junior"
    if years_experience < 7:
        return "mid"
    return "senior"


def matches_seniority(resume_seniority: str, job_seniority: str) -> bool:
    """Casefold comparison of seniority strings (strata are exact in data)."""
    return normalize(resume_seniority).lower() == normalize(job_seniority).lower()


def matches_industry(resume_industry: str, job_industry: str) -> bool:
    """Casefold comparison of industry strings."""
    return normalize(resume_industry).lower() == normalize(job_industry).lower()


_DEGREES = (
    (re.compile(r"\bph\.?d\b|\bdoctorate\b"), "phd"),
    (re.compile(r"\bmaster(?:'s)?\b|\bmsc|m\.?s\.?\b"), "masters"),
    (
        re.compile(
            r"\bbachelor(?:'s)?\b|\bb(?:\.)?s(?:\.)?\b|\bba\b"
            r"|\bengineer(?:ing)?\b|\bb\.?tech\b"
        ),
        "bachelors",
    ),
    (re.compile(r"\bdiploma\b|\bassociate(?:'s)?\b"), "diploma"),
    (re.compile(r"\bhigh\s+school\b|\bhsc\b"), "high_school"),
)


def degree_level(education: str) -> str | None:
    """Map an education string to a coarse degree level (``None`` unknown)."""
    lowered = normalize(education).lower()
    for pattern, level in _DEGREES:
        if pattern.search(lowered):
            return level
    return None


def role_overlap(resume_role: str, job_title: str) -> bool:
    """Casefold equality between the canonical resume role and job title."""
    return normalize(resume_role).lower() == normalize(job_title).lower()
