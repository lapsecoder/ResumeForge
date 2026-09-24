"""Transient output schemas for local semantic matching.

Nothing returned by this module is persisted. Similarity values are
normalized to [0, 1] but are NOT probabilities, and do NOT prove skill
possession.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.semantic_matching.config import (
    MODEL_LICENSE,
    MODEL_SOURCE_DEFAULT,
    SEMANTIC_IMPLEMENTATION_VERSION,
)
from app.semantic_matching.similarity import SimilarityLevel


class SemanticItemComparison(BaseModel):
    """Similarity between one job unit and its best resume unit."""

    job_unit_name: str
    resume_unit_name: str | None = None
    category: str
    cosine: float = Field(..., ge=-1.0, le=1.0)
    similarity: float = Field(..., ge=0.0, le=1.0)
    level: SimilarityLevel


class SemanticMatchMetadata(BaseModel):
    """Non-sensitive facts about the model backing a result."""

    method: str = "local-semantic-embedding"
    model_name: str
    model_version: str
    model_source: str = MODEL_SOURCE_DEFAULT
    model_license: str = MODEL_LICENSE
    model_dimension: int
    device: str
    high_similarity_threshold: float
    moderate_similarity_threshold: float
    implementation_version: str = SEMANTIC_IMPLEMENTATION_VERSION


class SemanticMatchResult(BaseModel):
    """Per-category semantic similarities plus item-level comparisons.

    ``*_similarity`` fields are None when the corresponding category has no
    comparable units on either side; the overall score averages only the
    present categories (no penalty for absent sections).
    """

    overall_similarity: float | None = Field(
        description="Mean of present categories, in [0, 1]."
    )
    overall_cosine: float | None = Field(
        description="Raw cosine form of overall_similarity."
    )
    summary_similarity: float | None = None
    skill_similarity: float | None = None
    experience_similarity: float | None = None
    responsibility_similarity: float | None = None
    project_similarity: float | None = None
    qualification_similarity: float | None = None

    matched_semantic_items: list[SemanticItemComparison] = Field(default_factory=list)
    related_items: list[SemanticItemComparison] = Field(default_factory=list)
    low_similarity_items: list[SemanticItemComparison] = Field(default_factory=list)

    note: str
    metadata: SemanticMatchMetadata


EMPTY_MATCH_NOTE = (
    "Semantic matching needs comparable content on both sides. "
    "Add a summary, skills, experience, projects, or education to get a result."
)

RESULT_NOTE = (
    "Semantic similarity is a measure of relatedness, not a hiring probability. "
    "High relatedness does not prove the candidate possesses a skill or meets a "
    "requirement; use deterministic skill matching for possession checks."
)
