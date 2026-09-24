"""Local semantic matching service (Phase 5B).

Combines PII-free text units, local embeddings, and cosine similarity into a
transient :class:`SemanticMatchResult`. Nothing is persisted; no external
service is contacted.
"""

from __future__ import annotations

from app.job_parsing.schemas import JobDescription
from app.parsing.schemas import Resume
from app.semantic_matching.config import semantic_settings
from app.semantic_matching.embedder import Embedder
from app.semantic_matching.model import (
    EmbeddingProvider,
    InferenceError,
    ModelMetadata,
    get_embedding_provider,
)
from app.semantic_matching.schemas import (
    EMPTY_MATCH_NOTE,
    RESULT_NOTE,
    SemanticItemComparison,
    SemanticMatchMetadata,
    SemanticMatchResult,
)
from app.semantic_matching.similarity import (
    cosine_similarity,
    normalized_similarity,
    similarity_level,
)
from app.semantic_matching.text_builder import (
    SemanticUnit,
    build_job_units,
    build_resume_units,
)

#: Allowed comparable resume categories for each job-unit category.
JOB_UNIT_POOLS: dict[str, tuple[str, ...]] = {
    "summary": ("summary",),
    "skill": ("skill",),
    "experience": ("experience",),
    "responsibility": ("experience", "project"),
    "qualification": ("education", "certification"),
    "education": ("education", "certification"),
    "certification": ("education", "certification"),
}

#: (result field, job categories, resume categories) for each reported metric.
_CATEGORY_LAYOUT: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    ("summary_similarity", ("summary",), ("summary",)),
    ("skill_similarity", ("skill",), ("skill",)),
    ("experience_similarity", ("experience",), ("experience",)),
    ("responsibility_similarity", ("responsibility",), ("experience",)),
    ("project_similarity", ("responsibility",), ("project",)),
    (
        "qualification_similarity",
        ("qualification", "education", "certification"),
        ("education", "certification"),
    ),
]


def compute_semantic_match(
    resume: Resume,
    job: JobDescription,
    *,
    provider: EmbeddingProvider | None = None,
) -> SemanticMatchResult:
    """Compute a transient semantic match between a resume and a job.

    The ``provider`` argument lets tests inject a fake embedding provider; when
    omitted, the process-wide singleton provider is used.
    """
    resume_units = build_resume_units(resume)
    job_units = build_job_units(job)

    if not resume_units or not job_units:
        return _empty_result()

    embedder = Embedder(provider or get_embedding_provider())
    resume_vectors = embedder.embed_units(resume_units)
    job_vectors = embedder.embed_units(job_units)

    if not resume_vectors or not job_vectors:
        return _empty_result()

    _validate_dimensions(resume_vectors, job_vectors)

    resultant: dict[str, float | None] = {}
    for field_name, job_cats, resume_cats in _CATEGORY_LAYOUT:
        resultant[field_name] = _category_similarity(
            job_units, resume_units, job_vectors, resume_vectors, job_cats, resume_cats
        )

    overall = _overall(resultant)
    metadata = _metadata_from(embedder.metadata())
    items = _item_comparisons(job_units, resume_units, job_vectors, resume_vectors)
    return SemanticMatchResult(
        overall_similarity=overall,
        overall_cosine=(
            _cosine_from_normalized(overall) if overall is not None else None
        ),
        summary_similarity=resultant["summary_similarity"],
        skill_similarity=resultant["skill_similarity"],
        experience_similarity=resultant["experience_similarity"],
        responsibility_similarity=resultant["responsibility_similarity"],
        project_similarity=resultant["project_similarity"],
        qualification_similarity=resultant["qualification_similarity"],
        matched_semantic_items=[
            item for item in items if item.level == "high"
        ],
        related_items=[item for item in items if item.level == "moderate"],
        low_similarity_items=[item for item in items if item.level == "low"],
        note=RESULT_NOTE,
        metadata=metadata,
    )


def _category_similarity(
    job_units: list[SemanticUnit],
    resume_units: list[SemanticUnit],
    job_vectors: dict[str, list[float]],
    resume_vectors: dict[str, list[float]],
    job_categories: tuple[str, ...],
    resume_categories: tuple[str, ...],
) -> float | None:
    """Mean over job units of the best normalized similarity vs. resume units.

    Returns None when either side has no comparable units.
    """
    job_selected = [unit for unit in job_units if unit.category in job_categories]
    resume_selected = [
        unit for unit in resume_units if unit.category in resume_categories
    ]
    if not job_selected or not resume_selected:
        return None
    total = 0.0
    for job_unit in job_selected:
        best = 0.0
        for resume_unit in resume_selected:
            cosine = cosine_similarity(
                job_vectors[job_unit.name], resume_vectors[resume_unit.name]
            )
            best = max(best, normalized_similarity(cosine))
        total += best
    return total / len(job_selected)


def _item_comparisons(
    job_units: list[SemanticUnit],
    resume_units: list[SemanticUnit],
    job_vectors: dict[str, list[float]],
    resume_vectors: dict[str, list[float]],
) -> list[SemanticItemComparison]:
    """One best-comparison per job unit, bucketed later by the caller."""
    comparisons: list[SemanticItemComparison] = []
    for job_unit in job_units:
        pool = JOB_UNIT_POOLS.get(job_unit.category, ())
        candidates = [
            unit
            for unit in resume_units
            if unit.category in pool and unit.name in resume_vectors
        ]
        if not candidates:
            continue
        best_resume_unit = max(
            candidates,
            key=lambda unit: cosine_similarity(
                job_vectors[job_unit.name], resume_vectors[unit.name]
            ),
        )
        cosine = cosine_similarity(
            job_vectors[job_unit.name], resume_vectors[best_resume_unit.name]
        )
        normalized = normalized_similarity(cosine)
        comparisons.append(
            SemanticItemComparison(
                job_unit_name=job_unit.name,
                resume_unit_name=best_resume_unit.name,
                category=best_resume_unit.category,
                cosine=cosine,
                similarity=normalized,
                level=similarity_level(
                    normalized,
                    high=semantic_settings.high_similarity_threshold,
                    moderate=semantic_settings.moderate_similarity_threshold,
                ),
            )
        )
    return comparisons


def _validate_dimensions(
    resume_vectors: dict[str, list[float]],
    job_vectors: dict[str, list[float]],
) -> None:
    dimensions = {
        len(vector)
        for vector in list(resume_vectors.values()) + list(job_vectors.values())
    }
    if len(dimensions) > 1 or 0 in dimensions:
        raise InferenceError("Embedding vectors have inconsistent dimensions.")


def _overall(resultant: dict[str, float | None]) -> float | None:
    present = [value for value in resultant.values() if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _cosine_from_normalized(normalized: float) -> float:
    return max(-1.0, min(1.0, normalized * 2.0 - 1.0))


def _metadata_from(metadata: ModelMetadata) -> SemanticMatchMetadata:
    # model_version falls back to the model_name when the provider does not
    # pin a revision: the repo id is still the model's version identity.
    model_version = metadata.model_version or metadata.model_name
    return SemanticMatchMetadata(
        model_name=metadata.model_name,
        model_version=model_version,
        model_source=metadata.source,
        model_license=metadata.license,
        model_dimension=metadata.model_dimension,
        device=metadata.device,
        high_similarity_threshold=semantic_settings.high_similarity_threshold,
        moderate_similarity_threshold=semantic_settings.moderate_similarity_threshold,
    )


def _empty_result() -> SemanticMatchResult:
    return SemanticMatchResult(
        overall_similarity=None,
        overall_cosine=None,
        note=EMPTY_MATCH_NOTE,
        metadata=SemanticMatchMetadata(
            model_name=semantic_settings.model_name,
            model_version=semantic_settings.model_name,
            model_dimension=semantic_settings.model_dimension,
            device="not-loaded",
            high_similarity_threshold=semantic_settings.high_similarity_threshold,
            moderate_similarity_threshold=semantic_settings.moderate_similarity_threshold,
        ),
    )
