"""Transient Pydantic schemas for general ATS Readiness / resume-quality analysis.

These models exist only for the lifetime of a request. Nothing is written to a
database or the filesystem, nothing is logged, and the input Resume is never
modified or retained.

The ATS Readiness Score is a deterministic, explainable heuristic that says
nothing about any specific commercial ATS. See ``rules.py`` for the documented
weights and thresholds and the disclaimer in ``AnalysisMetadata``.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Severity(str, Enum):
    """Deterministic finding severity, never derived from a model output."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Finding(BaseModel):
    """A single deterministic, rule-derived observation.

    Every finding is traceable to exactly one rule (``rule_id``). Findings
    whose ``impact`` is greater than zero caused a score deduction; zero-impact
    findings are advisory/informational only.
    """

    category: str
    severity: Severity
    rule_id: str
    title: str
    explanation: str
    evidence: str = Field(default="")
    recommendation: str
    impact: float = Field(
        default=0.0,
        ge=0.0,
        description="Points deducted for this finding (>0 only).",
    )


class CategoryScore(BaseModel):
    """Per-category score plus the weight that was actually applied.

    ``score`` is ``None`` when the category is not applicable to the resume
    (for example a resume with no work experience has no Experience Quality
    score); in that case its weight is redistributed to the other categories
    instead of penalising the resume.
    """

    key: str
    label: str
    score: float | None = Field(default=None, ge=0.0, le=100.0)
    applicable: bool
    weight: float = Field(
        description=(
            "Applied weight after redistribution (0 when not applicable)."
        )
    )
    max_weight: float = Field(
        description=(
            "Documented default weight before redistribution."
        )
    )


class AnalysisMetadata(BaseModel):
    """Version, weights, and scope disclaimer for the analysis run."""

    method: str
    version: str
    weights: dict[str, float]
    applied_weights: dict[str, float]
    score_labels: dict[str, str]
    disclaimer: str


class ATSReadinessResult(BaseModel):
    """Deterministic ATS Readiness / resume-quality analysis result."""

    overall_score: float | None = Field(default=None, ge=0.0, le=100.0)
    score_label: str | None
    category_scores: list[CategoryScore]
    findings: list[Finding]
    metadata: AnalysisMetadata
