"""Transient Pydantic schemas for the deterministic matching result.

These models exist only for the lifetime of a matching call. They are not
database tables, are not persisted, and are garbage-collected after the
response is sent. All optional/unknown evidence is represented explicitly.
"""

from __future__ import annotations

import datetime as _dt

from pydantic import BaseModel, Field

SCORE_WEIGHTS_DEFAULT: dict[str, float] = {
    "required_skill": 50.0,
    "preferred_skill": 15.0,
    "experience": 20.0,
    "education": 10.0,
    "qualification": 5.0,
}

BASELINE_NOTE = (
    "Deterministic heuristic baseline matching. Semantic similarity and "
    "embeddings are intentionally deferred to a later phase."
)


class SkillMatch(BaseModel):
    """Skill comparison between the resume and the job description."""

    score: float | None = Field(
        default=None,
        description="Composite skill score (0-100) or None when the job lists "
        "no skill requirements at all.",
    )
    matched_required: list[str] = Field(default_factory=list)
    matched_preferred: list[str] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    missing_preferred: list[str] = Field(default_factory=list)
    matched: list[str] = Field(
        default_factory=list, description="Required then preferred, deduplicated."
    )
    total_required: int = Field(default=0, ge=0)
    total_preferred: int = Field(default=0, ge=0)
    required_coverage: float | None = Field(
        default=None, description="0.0-1.0 or None when no required skills."
    )
    preferred_coverage: float | None = Field(
        default=None, description="0.0-1.0 or None when no preferred skills."
    )
    notes: list[str] = Field(default_factory=list)


class ExperienceMatch(BaseModel):
    """Experience comparison based on structured date ranges only."""

    score: float | None = Field(
        default=None,
        description="0-100, or None when not required or not verifiable.",
    )
    candidate_years: float | None = Field(
        default=None,
        description="Approximate total experience in years, or None when the "
        "resume date ranges are insufficient.",
    )
    required_years: float | None = Field(
        default=None, description="Explicit minimum years requested, or None."
    )
    required_max_years: float | None = Field(
        default=None, description="Upper bound of a requested range, or None."
    )
    required_level: str | None = Field(
        default=None,
        description="Broad seniority label (entry/junior/mid/senior) when stated.",
    )
    roles_total: int = Field(default=0, ge=0)
    roles_measured: int = Field(default=0, ge=0)
    assessment: str = Field(default="")


class EducationMatch(BaseModel):
    """Education comparison at the level of explicitly recognised degrees."""

    score: float | None = Field(default=None, description="0-100 or None.")
    required_level: str | None = Field(
        default=None, description="doctorate/master/bachelor/diploma or None."
    )
    candidate_level: str | None = Field(
        default=None, description="Highest recognised resume degree level, or None."
    )
    matched: bool | None = Field(
        default=None, description="True/False, or None when not required."
    )
    assessment: str = Field(default="")


class QualificationMatch(BaseModel):
    """Qualification comparison with explicit evidence limitations.

    Three distinct buckets:
    - matched  -> deterministically verified against the structured resume.
    - unmet    -> an explicit requirement that the resume does not list.
    - unknown  -> present in the JD but not verifiable from structured data
                  (absence of evidence is NOT treated as absence of the
                  qualification).
    """

    score: float | None = Field(default=None, description="0-100 or None.")
    matched: list[str] = Field(default_factory=list)
    unmet: list[str] = Field(default_factory=list)
    unknown: list[str] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    assessment: str = Field(default="")


class MatchMetadata(BaseModel):
    method: str = Field(default="deterministic-baseline")
    label: str = Field(default="Baseline Match Score")
    version: str = Field(default="5a-baseline-1.0")
    weights: dict[str, float] = Field(
        default_factory=lambda: dict(SCORE_WEIGHTS_DEFAULT)
    )
    applied_weights: dict[str, float] = Field(
        default_factory=dict, description="Weights used after redistribution."
    )
    reference_date: _dt.date = Field(
        description="Date used to resolve 'Present' experience end dates."
    )
    note: str = Field(default=BASELINE_NOTE)


class MatchResult(BaseModel):
    """Deterministic, explainable baseline match between Resume and JobDescription."""

    overall_score: float | None = Field(
        default=None,
        description="Weighted baseline match score (0-100), or None when the job "
        "defines no evaluable requirements.",
    )
    skill_match: SkillMatch = Field(default_factory=SkillMatch)
    experience_match: ExperienceMatch = Field(default_factory=ExperienceMatch)
    education_match: EducationMatch = Field(default_factory=EducationMatch)
    qualification_match: QualificationMatch = Field(default_factory=QualificationMatch)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    matched_requirements: list[str] = Field(default_factory=list)
    unmet_requirements: list[str] = Field(default_factory=list)
    metadata: MatchMetadata
