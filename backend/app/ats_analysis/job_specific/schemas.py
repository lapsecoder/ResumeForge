"""Transient Pydantic schemas for Job-Specific ATS Coverage analysis.

These models exist only for the lifetime of a request. Nothing is written to a
database or the filesystem, nothing is logged, and the input Resume and
JobDescription are never modified or retained.

The score is a deterministic, explainable heuristic that measures terminology
representation, not a hiring/ATS-pass probability. See ``rules.py`` for the
documented weights and thresholds and the disclaimer in
``AxisAnalysisMetadata``.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.ats_analysis.schemas import AnalysisMetadata, CategoryScore, Finding


class MatchType(str, Enum):
    """How a job term was located in the resume, from strongest to weakest.

    - ``exact``: the term's literal normalised form appears as a resume skill.
    - ``normalized``: the term matches a resume skill via canonicalisation
      (case, unicode, resume-side alias expansion).
    - ``alias``: the job term itself is a known abbreviation (e.g. ``js``)
      whose canonical form appears as a resume skill.
    - ``phrase``: the term was found in resume free text (token-aware).
    - ``absent``: the term was not found anywhere in the resume.
    """

    EXACT = "exact"
    NORMALIZED = "normalized"
    ALIAS = "alias"
    PHRASE = "phrase"
    ABSENT = "absent"


class TermOrigin(str, Enum):
    """Where the matched job term came from in the job description."""

    REQUIRED = "required"
    PREFERRED = "preferred"
    PHRASE = "phrase"


class TermMatch(BaseModel):
    """One job-description term and how it was (or was not) found."""

    term: str
    origin: TermOrigin
    match_type: MatchType
    evidence_locations: list[str] = Field(default_factory=list)
    evidence: str = Field(default="", description="Short locator snippet.")
    explanation: str = Field(default="", description="Why this result was produced.")


class CoverageTotals(BaseModel):
    """Counts per terminology bucket, all deterministic."""

    required: int = Field(..., ge=0)
    required_matched: int = Field(..., ge=0)
    preferred: int = Field(..., ge=0)
    preferred_matched: int = Field(..., ge=0)
    phrases: int = Field(..., ge=0)
    phrases_matched: int = Field(..., ge=0)


class CoverageReport(BaseModel):
    """Coverage fractions plus aggregate counts.

    A fraction is ``None`` when the corresponding bucket is empty (never a
    0: absence of a bucket is not treated as zero coverage).
    """

    required_coverage: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Matched required terms / total required terms.",
    )
    preferred_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    overall_coverage: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="(required + preferred) matched over total required + preferred.",
    )
    evidence_supported_required: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description=(
            "Matched required terms with experience or project evidence, over "
            "all matched required terms."
        ),
    )
    evidence_supported_preferred: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Same as evidence_supported_required for preferred terms.",
    )

    totals: CoverageTotals


class JobSpecificATSResult(BaseModel):
    """Deterministic Job-Specific ATS Coverage analysis result."""

    overall_score: float | None = Field(default=None, ge=0.0, le=100.0)
    score_label: str | None = None
    category_scores: list[CategoryScore]
    coverage: CoverageReport
    term_matches: list[TermMatch]
    findings: list[Finding]
    metadata: AnalysisMetadata
