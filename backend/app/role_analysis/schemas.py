"""Transient Pydantic schemas for role-level analysis.

These models exist only for the lifetime of a role analysis call.
Nothing is persisted. The profile data is local, deterministic, and
derived from the local knowledge base in ``profiles.py``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.job_parsing.schemas import JobDescription
from app.matching.schemas import MatchResult
from app.parsing.schemas import Resume


class RoleProfileInfo(BaseModel):
    """Metadata about the resolved role profile."""

    title: str
    aliases: list[str] = Field(default_factory=list)
    supported: bool = True


class SkillComparison(BaseModel):
    """Comparison of resume skills against role requirements."""

    have: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    matched_count: int = 0
    required_total: int = 0
    preferred_total: int = 0


class ExperienceComparison(BaseModel):
    """Comparison of resume experience against role expectations."""

    candidate_years: float | None = None
    required_years: float | None = None
    required_level: str | None = None
    gap_years: float | None = None
    met: bool | None = None
    resume_experience_count: int = 0


class EducationComparison(BaseModel):
    """Comparison of resume education against role expectations."""

    required_level: str | None = None
    candidate_level: str | None = None
    met: bool | None = None


class RoleAnalysisResult(BaseModel):
    """Result of role-level compatibility analysis.

    This is distinct from ``HybridMatchResult`` (which is
    JD-specific). Role-level analysis uses the local profile's
    structured expectations; JD-specific analysis uses the user's
    pasted/uploaded job description.
    """

    role_title: str
    profile: RoleProfileInfo
    compatibility: dict[str, float | None] = Field(default_factory=dict)
    skills: SkillComparison = Field(default_factory=SkillComparison)
    experience: ExperienceComparison = Field(default_factory=ExperienceComparison)
    education: EducationComparison = Field(default_factory=EducationComparison)
    matched_requirements: list[str] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    matched_job: JobDescription | None = Field(
        default=None,
        description="The JobDescription derived from the profile, if any.",
    )
    deterministic_match: MatchResult | None = Field(
        default=None,
        description="The deterministic matching result, if profile had requirements.",
    )


class RoleAnalysisRequest(BaseModel):
    """Request for role-level compatibility analysis."""

    role_title: str = Field(..., description="Target job role title.")
    resume: Resume = Field(..., description="Parsed resume to analyze.")
