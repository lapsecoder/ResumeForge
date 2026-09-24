r"""Optional integration test running the REAL local embedding model.

Skipped when sentence-transformers is missing or cannot be used on the current
machine (e.g. OS policies blocking native DLLs). Run manually:

    $env:PYTHONIOENCODING="utf-8"
    & .venv\Scripts\python.exe -m pytest tests/test_semantic_integration.py
        -m integration -s

This is the only test in the suite that exercises the actual model.
"""

from __future__ import annotations

import pytest

try:
    import sentence_transformers  # noqa: F401, PLC0415
except Exception as exc:  # missing package, blocked native DLL, etc.
    pytest.skip(
        f"sentence-transformers is unavailable on this machine: {exc}",
        allow_module_level=True,
    )

from app.job_parsing.schemas import JobDescription, JobMetadata  # noqa: E402, PLC0415
from app.parsing.schemas import (  # noqa: E402, PLC0415
    ConfidenceLevel,
    Resume,
    ResumeMetadata,
    SkillSet,
)
from app.semantic_matching.model import (  # noqa: E402, PLC0415
    LocalSentenceTransformerProvider,
    ModelLoadError,
)
from app.semantic_matching.service import compute_semantic_match  # noqa: E402, PLC0415

pytestmark = pytest.mark.integration


def _load_provider() -> LocalSentenceTransformerProvider:
    provider = LocalSentenceTransformerProvider()
    try:
        provider.embed_texts(["warmup"])
    except ModelLoadError as exc:
        pytest.skip(f"Model could not be loaded: {exc}")
    return provider


def test_default_model_produces_finite_384_dim_vectors() -> None:
    provider = _load_provider()
    vectors = provider.embed_texts(
        ["Python backend engineering", "Baking sourdough bread"]
    )
    assert len(vectors) == 2
    assert all(len(vector) == 384 for vector in vectors)
    assert all(all(v == v for v in vector) for vector in vectors)  # no NaN
    metadata = provider.metadata()
    assert metadata.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert metadata.device in ("cpu", "cuda")


def test_semantically_related_exceeds_unrelated() -> None:
    provider = _load_provider()
    resume = Resume(
        summary="Backend engineer building Python services.",
        skills=SkillSet(all=["Python", "PostgreSQL"]),
        metadata=ResumeMetadata(
            word_count=0, file_type="pdf", overall_confidence=ConfidenceLevel.HIGH
        ),
    )
    related_job = JobDescription(
        title="Backend Engineer",
        summary="Design Python services on PostgreSQL.",
        required_skills=["Python"],
        metadata=JobMetadata(word_count=0, overall_confidence=ConfidenceLevel.HIGH),
    )
    unrelated_job = JobDescription(
        title="Pastry Chef",
        summary="Bake sourdough bread and manage a kitchen team.",
        required_skills=["Baking"],
        metadata=JobMetadata(word_count=0, overall_confidence=ConfidenceLevel.HIGH),
    )
    related = compute_semantic_match(resume, related_job, provider=provider)
    unrelated = compute_semantic_match(resume, unrelated_job, provider=provider)
    assert related.overall_similarity is not None
    assert unrelated.overall_similarity is not None
    assert related.overall_similarity > unrelated.overall_similarity
