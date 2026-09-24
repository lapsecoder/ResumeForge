"""Deterministic experience matching.

Candidate experience is computed ONLY from structured ``WorkExperience``
date ranges. Missing or unparseable dates are never invented: entries without
both a start and end date are excluded from the total and surfaced in the
assessment. When candidate years cannot be derived at all the result is
explicitly ``unknown`` rather than zero.

Required years are parsed from the job's ``experience_requirements``
statements (``2+ years``, ``minimum 5 years``, ``3-5 years``, ``Senior level``).
Months are essentially year boundaries: ``MM/YYYY`` carries a month,
``Month YYYY`` carries a month, bare ``YYYY`` is treated as June of that year
(neutral midpoint) so single-year stints add roughly one year.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass

from app.job_parsing.schemas import JobDescription
from app.matching.schemas import ExperienceMatch
from app.parsing.schemas import Resume

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_MONTH_NAME = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"

_MM_YYYY_RE = re.compile(r"^\s*(\d{1,2})\s*/\s*(\d{4})\s*$")
_MONTH_YEAR_RE = re.compile(rf"^\s*({_MONTH_NAME})\s+(\d{{4}})\s*$", re.IGNORECASE)
_YEAR_RE = re.compile(r"^\s*(\d{4})\s*$")

_REQUIRED_YEARS_RANGE_RE = re.compile(
    r"\b(\d{1,2}(?:\.\d)?)\s*(?:-|–|—|to)\s*(\d{1,2}(?:\.\d)?)\s*\+?\s*y(?:ea)?rs?\b",
    re.IGNORECASE,
)
_REQUIRED_YEARS_NUM_RE = re.compile(
    r"\b(\d{1,2}(?:\.\d)?)\s*\+?\s*y(?:ea)?rs?\b", re.IGNORECASE
)
_REQUIRED_YEARS_PREFIX_RE = re.compile(
    r"\b(?:minimum|at\s*least|min\.?)\s*\d", re.IGNORECASE
)

_LEVEL_RE = re.compile(
    r"\b(entry[- ]level|fresher|junior[- ]level|junior|mid[- ]level|mid\b|"
    r"senior[- ]level|senior)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _YearMonth:
    year: int
    month: int | None = None

    def index(self) -> int:
        return self.year * 12 + (self.month if self.month is not None else 6)


def parse_experience_date(value: str | None, reference: _dt.date) -> _YearMonth | None:
    if not value:
        return None
    text = value.strip()
    lowered = text.lower()
    if lowered in {"present", "current", "now"}:
        return _YearMonth(reference.year, reference.month)

    match = _MM_YYYY_RE.fullmatch(text)
    if match:
        month, year = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            return _YearMonth(year, month)
        return None

    match = _MONTH_YEAR_RE.fullmatch(text)
    if match:
        name, year = match.group(1), int(match.group(2))
        month_num = _MONTHS.get(name[:3].lower())
        if month_num is not None:
            return _YearMonth(year, month_num)
        return None

    match = _YEAR_RE.fullmatch(text)
    if match:
        return _YearMonth(int(match.group(1)))
    return None


def candidate_experience_years(
    resume: Resume, reference: _dt.date
) -> tuple[float | None, int, int]:
    """Return (years | None, measured_roles, total_roles)."""
    total = 0.0
    measured = 0
    for entry in resume.experience:
        start = parse_experience_date(entry.start_date, reference)
        end = parse_experience_date(entry.end_date, reference)
        if start is None or end is None:
            continue
        months = end.index() - start.index()
        if months <= 0:
            continue
        total += months / 12
        measured += 1
    years = round(total, 1) if measured else None
    return years, measured, len(resume.experience)


@dataclass(frozen=True)
class _RequiredYears:
    minimum: float | None
    maximum: float | None
    level: str | None


def parse_required_experience(job: JobDescription) -> _RequiredYears:
    minimum: float | None = None
    maximum: float | None = None
    level: str | None = None

    for statement in job.experience_requirements:
        text = statement.strip()
        if not text:
            continue
        match = _LEVEL_RE.search(text)
        if match:
            token = match.group(0).lower().replace(" ", "")
            if "entry" in token or "fresher" in token:
                level = "entry"
            elif "junior" in token:
                level = "junior"
            elif "mid" in token:
                level = "mid"
            elif "senior" in token:
                level = "senior"

        range_match = _REQUIRED_YEARS_RANGE_RE.search(text)
        if range_match:
            low, high = float(range_match.group(1)), float(range_match.group(2))
            if low > high:
                low, high = high, low
            minimum = max(minimum, low) if minimum is not None else low
            maximum = max(maximum, high) if maximum is not None else high
            continue

        num_match = _REQUIRED_YEARS_NUM_RE.search(text)
        if not num_match:
            continue
        years = float(num_match.group(1))
        prefix = bool(_REQUIRED_YEARS_PREFIX_RE.search(text))
        minimum_markers = ("+" in text) or ("minimum" in text.lower())
        minimum_markers = minimum_markers or ("least" in text.lower())
        if prefix or minimum_markers:
            minimum = max(minimum, years) if minimum is not None else years
        elif minimum is None:
            minimum = years

    return _RequiredYears(minimum=minimum, maximum=maximum, level=level)


def match_experience(
    resume: Resume, job: JobDescription, reference: _dt.date
) -> ExperienceMatch:
    candidate, roles_measured, roles_total = candidate_experience_years(
        resume, reference
    )
    required = parse_required_experience(job)

    if required.minimum is None and required.level is None:
        return ExperienceMatch(
            score=None,
            candidate_years=candidate,
            required_years=None,
            required_level=None,
            roles_total=roles_total,
            roles_measured=roles_measured,
            assessment=(
                f"Job states no explicit experience requirement — not evaluated. "
                f"Resume provides {roles_measured} of {roles_total} measurable "
                f"experience entries."
            ),
        )

    if candidate is None:
        return ExperienceMatch(
            score=None,
            candidate_years=None,
            required_years=required.minimum,
            required_max_years=required.maximum,
            required_level=required.level,
            roles_total=roles_total,
            roles_measured=roles_measured,
            assessment=(
                "Experience requirement could not be verified: resume date ranges "
                "are insufficient to calculate candidate experience. This is NOT "
                "treated as zero experience."
            ),
        )

    base_assessment = (
        f"Candidate has ~{candidate} years from {roles_measured} of {roles_total} "
        f"measurable experience entries; "
    )

    if required.minimum is None:
        return ExperienceMatch(
            score=None,
            candidate_years=candidate,
            required_years=None,
            required_max_years=None,
            required_level=required.level,
            roles_total=roles_total,
            roles_measured=roles_measured,
            assessment=(
                base_assessment
                + f"job states a seniority level ({required.level}) without "
                "quantified years, so experience is not quantitatively scored."
            ),
        )

    required_years = required.minimum
    if candidate >= required_years:
        score = 100.0
        verdict = "requirement met."
    else:
        score = round(candidate / required_years * 100, 1)
        verdict = "requirement not met."

    return ExperienceMatch(
        score=score,
        candidate_years=candidate,
        required_years=required_years,
        required_max_years=required.maximum,
        required_level=required.level,
        roles_total=roles_total,
        roles_measured=roles_measured,
        assessment=(
            base_assessment
            + f"job requires {_fmt_years(required_years)}; "
            + verdict
        ),
    )


def _fmt_years(years: float) -> str:
    if years.is_integer():
        return f"{int(years)}+ years"
    return f"{years:g}+ years"
