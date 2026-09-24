"""Transient output schemas for the hybrid matching layer (Phase 5C).

Follows the docs/data-model.md conventions: these are in-memory Pydantic
models, never database tables, and they exist only for the lifetime of a
matching call. The result keeps the authoritative deterministic MatchResult
and the supporting semantic SemanticMatchResult distinct, and never exposes
raw embedding vectors.
"""

from __future__ import annotations

import datetime as _dt
from typing import Literal

from pydantic import BaseModel, Field

from app.matching.schemas import MatchResult
from app.semantic_matching.schemas import (
    SemanticMatchMetadata,
    SemanticMatchResult,
)
from app.semantic_matching.similarity import SimilarityLevel

HYBRID_LABEL = "Hybrid Match Score"
HYBRID_VERSION = "5c-hybrid-1.0"
HYBRID_NOTE = (
    "The Hybrid Match Score is a heuristic relevance score. It is not a "
    "hiring probability, an ATS probability, or an employment prediction. "
    "Deterministic evidence remains authoritative for explicit requirements; "
    "semantic similarity only supplements context-level matching."
)

HYBRID_MODEL_DISCLOSURE = (
    "The semantic signal is produced entirely locally by an open-source "
    "sentence-embedding model (Apache-2.0). No external, hosted, or paid AI "
    "model is consulted, and no content leaves the machine."
)

#: How the deterministic and semantic signals were combined.
HybridMatchMode = Literal[
    "hybrid",
    "deterministic-only",
    "semantic-only",
    "no-evidence",
]

#: How much structured information supported the result (NOT a confidence).
EvidenceQuality = Literal["high", "medium", "limited"]

#: Why a result is deterministic-only (semantic signal missing or unusable).
SemanticStatus = Literal[
    "available",
    "no_content",
    "model_unavailable",
    "inference_failed",
    "error",
]


class SemanticAvailability(BaseModel):
    """Whether the local semantic signal contributed to a hybrid result."""

    available: bool = Field(
        description="True when the semantic layer returned usable evidence."
    )
    status: SemanticStatus
    note: str = Field(
        default="",
        description="Non-sensitive reason when the semantic signal is missing.",
    )


class ComponentScores(BaseModel):
    """Per-signal scores in their native scales.

    ``hybrid_overall`` mirrors the top-level ``overall_score``. Semantic
    values are normalized to [0, 1]; deterministic values are 0-100. A value
    is ``None`` when the corresponding component was not evaluable (never a
    manufactured zero).
    """

    deterministic_overall: float | None = Field(default=None, description="0-100.")
    semantic_overall: float | None = Field(default=None, description="0-1.")
    hybrid_overall: float | None = Field(default=None, description="0-100.")
    deterministic_skills: float | None = Field(default=None, description="0-100.")
    semantic_skills: float | None = Field(default=None, description="0-1.")
    deterministic_experience: float | None = Field(default=None, description="0-100.")
    semantic_experience: float | None = Field(default=None, description="0-1.")
    deterministic_education: float | None = Field(default=None, description="0-100.")
    semantic_qualification: float | None = Field(default=None, description="0-1.")


class SemanticInsight(BaseModel):
    """A deterministic, traceable statement about semantic relatedness.

    Insights are rule-generated (never free-form LLM text) and explicitly
    distinguish "semantically related" from "explicitly matched". Similarity
    is the normalized [0, 1] value unless otherwise stated.
    """

    category: str
    evidence_level: SimilarityLevel
    similarity: float | None = None
    statement: str


class HybridMetadata(BaseModel):
    """Non-sensitive metadata describing how a hybrid result was produced."""

    method: str = "hybrid-match"
    label: str = HYBRID_LABEL
    version: str = HYBRID_VERSION
    mode: HybridMatchMode
    weights: dict[str, float] = Field(
        default_factory=dict, description='{"deterministic": 0.70, "semantic": 0.30}.'
    )
    evidence_quality: EvidenceQuality
    semantic_availability: SemanticAvailability
    semantic_metadata: SemanticMatchMetadata | None = Field(
        default=None,
        description="Model facts from the semantic layer, or None when unused.",
    )
    reference_date: _dt.date = Field(
        description="Date used to resolve 'Present' experience end dates."
    )
    note: str = Field(default=HYBRID_NOTE)
    model_disclosure: str = Field(default=HYBRID_MODEL_DISCLOSURE)


class HybridMatchResult(BaseModel):
    """Transient, explainable hybrid match between a Resume and JobDescription."""

    overall_score: float | None = Field(
        default=None,
        description="Hybrid Match Score (0-100), or None when no meaningful "
        "hybrid evidence exists (e.g. semantic-only).",
    )
    deterministic: MatchResult
    semantic: SemanticMatchResult | None = Field(
        default=None,
        description="Supporting semantic result, or None when the model is "
        "unavailable or failed (the deterministic signal still stands).",
    )
    component_scores: ComponentScores = Field(default_factory=ComponentScores)
    matched_requirements: list[str] = Field(
        default_factory=list, description="Deterministic explicit matches only."
    )
    missing_required: list[str] = Field(
        default_factory=list,
        description="Deterministic required skills not explicitly verified.",
    )
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    semantic_insights: list[SemanticInsight] = Field(default_factory=list)
    metadata: HybridMetadata
