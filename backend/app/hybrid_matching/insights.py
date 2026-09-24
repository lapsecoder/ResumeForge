"""Deterministic semantic-insight generation for hybrid results (Phase 5C).

Insights are produced by fixed rules, never by an LLM. Every insight speaks
about *relatedness* — evidence of context-level fit — and never upgrades
relatedness into verified possession. Explicit matches continue to live in the
deterministic result (its matched/unmet lists and strengths/gaps); this module
only supplements that authoritative output.
"""

from __future__ import annotations

from app.hybrid_matching.schemas import SemanticInsight
from app.job_parsing.schemas import JobDescription
from app.matching.normalizer import normalize_skill
from app.matching.schemas import MatchResult
from app.semantic_matching.config import semantic_settings
from app.semantic_matching.schemas import (
    SemanticItemComparison,
    SemanticMatchResult,
)
from app.semantic_matching.similarity import SimilarityLevel, similarity_level

RELATEDNESS_CAVEAT = (
    "Semantic relatedness supports context-level matching but does not prove "
    "explicit requirement satisfaction."
)

#: (result field, insight category, resume-side label, job-side label) for
#: category-level insights. Skill and qualification relatedness are covered by
#: item-level and category-level rules below.
_CATEGORY_CONFIG: list[tuple[str, str, str, str]] = [
    ("summary_similarity", "summary", "summary", "summary"),
    (
        "experience_similarity",
        "experience",
        "experience",
        "experience requirement",
    ),
    (
        "responsibility_similarity",
        "responsibility",
        "experience",
        "responsibilities",
    ),
    ("project_similarity", "project", "project", "responsibilities"),
    (
        "qualification_similarity",
        "qualification",
        "education/certification",
        "qualification and education requirements",
    ),
]

_STRENGTH: dict[SimilarityLevel, str] = {
    "high": "strong",
    "moderate": "moderate",
    "low": "limited",
}


def build_semantic_insights(
    deterministic: MatchResult,
    semantic: SemanticMatchResult | None,
    job: JobDescription,
) -> list[SemanticInsight]:
    """Generate deterministic, traceable semantic insights.

    An empty list means either that no semantic evidence exists or that none
    of it crossed the insight thresholds.
    """
    if semantic is None:
        return []

    insights: list[SemanticInsight] = []

    # Item-level: missing required/preferred skills with supporting evidence.
    for kind, skills, matched in (
        ("required", job.required_skills, deterministic.skill_match.matched_required),
        (
            "preferred",
            job.preferred_skills,
            deterministic.skill_match.matched_preferred,
        ),
    ):
        for index, skill in enumerate(skills):
            if not skill.strip() or _explicitly_matched(skill, matched):
                continue
            item = _item_for(semantic, f"{kind}-skill:{index}")
            if item is None or item.level not in ("high", "moderate"):
                continue
            insights.append(
                SemanticInsight(
                    category="skill",
                    evidence_level=item.level,
                    similarity=round(item.similarity, 3),
                    statement=(
                        f"{kind.capitalize()} skill '{skill.strip()}' is not "
                        "explicitly verified: semantic evidence may indicate "
                        "related experience, but does not establish possession "
                        f"(relatedness {item.similarity:.2f})."
                    ),
                )
            )

    # Category-level: high/moderate relatedness between resume and job content.
    for field_name, category, resume_label, job_label in _CATEGORY_CONFIG:
        similarity = _category_similarity(semantic, field_name)
        if similarity is None:
            continue
        level = similarity_level(
            similarity,
            high=semantic_settings.high_similarity_threshold,
            moderate=semantic_settings.moderate_similarity_threshold,
        )
        if level == "low":
            continue
        insights.append(
            SemanticInsight(
                category=category,
                evidence_level=level,
                similarity=round(similarity, 3),
                statement=(
                    f"Resume {resume_label} content shows {_STRENGTH[level]} "
                    f"semantic relatedness to the job's {job_label} (normalized "
                    f"similarity {similarity:.2f}). {RELATEDNESS_CAVEAT}"
                ),
            )
        )

    # Overall relatedness, when it crossed the reporting bar.
    if semantic.overall_similarity is not None:
        level = similarity_level(
            semantic.overall_similarity,
            high=semantic_settings.high_similarity_threshold,
            moderate=semantic_settings.moderate_similarity_threshold,
        )
        if level in ("high", "moderate"):
            insights.append(
                SemanticInsight(
                    category="overall",
                    evidence_level=level,
                    similarity=round(semantic.overall_similarity, 3),
                    statement=(
                        f"Overall semantic relatedness between resume and job "
                        f"content is {_STRENGTH[level]} (normalized similarity "
                        f"{semantic.overall_similarity:.2f}). {RELATEDNESS_CAVEAT}"
                    ),
                )
            )

    return insights


def _explicitly_matched(skill: str, matched: list[str]) -> bool:
    """True when the deterministic layer already verified the skill."""
    norm = normalize_skill(skill)
    return any(normalize_skill(item) == norm for item in matched)


def _item_for(
    semantic: SemanticMatchResult, job_unit_name: str
) -> SemanticItemComparison | None:
    """Find an item comparison for a specific job unit, if one was produced."""
    for item in (
        *semantic.matched_semantic_items,
        *semantic.related_items,
        *semantic.low_similarity_items,
    ):
        if item.job_unit_name == job_unit_name:
            return item
    return None


def _category_similarity(
    semantic: SemanticMatchResult, field_name: str
) -> float | None:
    value = getattr(semantic, field_name)
    if value is None:
        return None
    return float(value)
