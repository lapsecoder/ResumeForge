"""Service tests for hybrid matching (Phase 5C) with controlled vectors.

The fake provider returns predictable vectors for every PII-free unit text, so
similarities are exact cosine math. No model library is required. These tests
verify the blend, the fallbacks, hard-requirement safety, alias consistency,
semantic insights, and component scores.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from app.hybrid_matching.service import compute_hybrid_match
from app.job_parsing.schemas import JobDescription, JobMetadata
from app.parsing.schemas import (
    Certification,
    ConfidenceLevel,
    Education,
    Project,
    Resume,
    ResumeMetadata,
    SkillSet,
    WorkExperience,
)
from app.semantic_matching.model import InferenceError, ModelLoadError, ModelMetadata
from app.semantic_matching.text_builder import build_job_units, build_resume_units

REFERENCE = date(2026, 9, 9)

IOTA = [1.0, 0.0]
ORTHOGONAL = [0.0, 1.0]
OPPOSITE = [-1.0, 0.0]


class VectorProvider:
    """Returns per-text vectors; raises lazily when told to."""

    def __init__(self, vectors: dict[str, list[float]], default: list[float] = IOTA):
        self._vectors = vectors
        self._default = default
        self._error: type[Exception] | None = None

    def fail_with(self, error: type[Exception]) -> "VectorProvider":
        self._error = error
        return self

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self._error is not None:
            raise self._error(f"injected {self._error.__name__}")
        return [self._vectors.get(text, self._default) for text in texts]

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(model_name="fake-hybrid", model_dimension=2, device="cpu")


def _resume_metadata() -> ResumeMetadata:
    return ResumeMetadata(
        word_count=0, file_type="pdf", overall_confidence=ConfidenceLevel.HIGH
    )


def _job_metadata() -> JobMetadata:
    return JobMetadata(word_count=0, overall_confidence=ConfidenceLevel.HIGH)


def make_resume(
    *,
    skills: list[str] | None = None,
    summary: str | None = None,
    experience: list[WorkExperience] | None = None,
    projects: list[Project] | None = None,
    education: list[Education] | None = None,
    certifications: list[Certification] | None = None,
) -> Resume:
    return Resume(
        summary=summary,
        skills=SkillSet(all=skills or []),
        experience=experience or [],
        projects=projects or [],
        education=education or [],
        certifications=certifications or [],
        metadata=_resume_metadata(),
    )


def make_job(
    *,
    summary: str | None = None,
    required: list[str] | None = None,
    preferred: list[str] | None = None,
    responsibilities: list[str] | None = None,
    experience_requirements: list[str] | None = None,
    education_requirements: list[str] | None = None,
    certifications: list[str] | None = None,
    qualifications: list[str] | None = None,
) -> JobDescription:
    return JobDescription(
        summary=summary,
        required_skills=required or [],
        preferred_skills=preferred or [],
        responsibilities=responsibilities or [],
        experience_requirements=experience_requirements or [],
        education_requirements=education_requirements or [],
        certifications=certifications or [],
        qualifications=qualifications or [],
        metadata=_job_metadata(),
    )


def _vectors_for(
    resume: Resume,
    job: JobDescription,
    *,
    job_vector: list[float] = IOTA,
) -> dict[str, list[float]]:
    """Resume units map to one vector, job units to another."""
    mapping: dict[str, list[float]] = {}
    for unit in build_resume_units(resume):
        mapping[unit.text] = list(IOTA)
    for unit in build_job_units(job):
        mapping[unit.text] = list(job_vector)
    return mapping


def _aligned_provider(resume: Resume, job: JobDescription) -> VectorProvider:
    """Everything collinear: every similarity is 1.0."""
    return VectorProvider(_vectors_for(resume, job), default=IOTA)


def _orthogonal_provider(resume: Resume, job: JobDescription) -> VectorProvider:
    """Job side orthogonal to the resume side: similarity is 0.5."""
    return VectorProvider(
        _vectors_for(resume, job, job_vector=ORTHOGONAL), default=IOTA
    )


def _opposite_provider(resume: Resume, job: JobDescription) -> VectorProvider:
    """Job side anti-aligned with the resume side: similarity is 0.0."""
    return VectorProvider(
        _vectors_for(resume, job, job_vector=OPPOSITE), default=IOTA
    )


def _hybrid(resume: Resume, job: JobDescription, provider: VectorProvider):
    return compute_hybrid_match(
        resume, job, reference_date=REFERENCE, provider=provider
    )


def stint(start: str, end: str | None) -> WorkExperience:
    return WorkExperience(
        company="Acme", title="Engineer", start_date=start, end_date=end
    )


class TestHybridBlending:
    def test_blends_7030_exactly(self) -> None:
        resume = make_resume(skills=["Python"])
        job = make_job(required=["Python", "Docker"])
        result = _hybrid(resume, job, _aligned_provider(resume, job))
        assert result.deterministic.overall_score == pytest.approx(50.0)
        assert result.semantic is not None
        assert result.semantic.overall_similarity == pytest.approx(1.0)
        assert result.overall_score == pytest.approx(65.0)
        assert result.metadata.mode == "hybrid"
        assert result.component_scores.deterministic_overall == pytest.approx(50.0)
        assert result.component_scores.semantic_overall == pytest.approx(1.0)
        assert result.component_scores.hybrid_overall == pytest.approx(65.0)

    def test_component_scores_are_native_scales(self) -> None:
        resume = make_resume(
            skills=["Python"],
            experience=[stint("2020", "2023")],
            education=[Education(institution="IIT", degree="B.Tech")],
        )
        job = make_job(
            required=["Python"],
            experience_requirements=["3+ years"],
            education_requirements=["Bachelor's degree"],
            qualifications=["Analytical thinker"],
        )
        result = _hybrid(resume, job, _aligned_provider(resume, job))
        score = result.component_scores
        assert score.deterministic_experience == pytest.approx(100.0)
        assert score.deterministic_education == pytest.approx(100.0)
        assert score.semantic_experience == pytest.approx(1.0)
        assert score.semantic_skills == pytest.approx(1.0)
        assert score.deterministic_skills == pytest.approx(100.0)
        assert result.metadata.evidence_quality == "high"

    def test_no_experience_requirement_is_not_penalised(self) -> None:
        resume = make_resume(skills=["Python"], experience=[stint("2020", "2023")])
        job = make_job(required=["Python"])
        result = _hybrid(resume, job, _aligned_provider(resume, job))
        assert result.component_scores.deterministic_experience is None
        assert result.component_scores.semantic_experience is None
        assert result.overall_score == pytest.approx(100.0)


class TestSemanticFallback:
    def test_model_unavailable_degrades_to_deterministic(self) -> None:
        resume = make_resume(skills=["Python"])
        job = make_job(required=["Python"])
        provider = _aligned_provider(resume, job).fail_with(ModelLoadError)
        result = _hybrid(resume, job, provider)
        assert result.semantic is None
        assert result.metadata.mode == "deterministic-only"
        assert result.overall_score == pytest.approx(100.0)
        assert result.metadata.semantic_availability.available is False
        assert result.metadata.semantic_availability.status == "model_unavailable"
        assert result.metadata.semantic_metadata is None
        assert "deterministic signal only" in result.metadata.semantic_availability.note

    def test_inference_failure_degrades_to_deterministic(self) -> None:
        resume = make_resume(skills=["Python"])
        job = make_job(required=["Python"])
        provider = _aligned_provider(resume, job).fail_with(InferenceError)
        result = _hybrid(resume, job, provider)
        assert result.semantic is None
        assert result.metadata.mode == "deterministic-only"
        assert result.metadata.semantic_availability.status == "inference_failed"

    def test_experience_matching_still_works_when_semantic_down(self) -> None:
        # I: the deterministic signal must never be a casualty of the model.
        resume = make_resume(
            skills=["Python"], experience=[stint("2020", "2021")]
        )
        job = make_job(required=["Python"], experience_requirements=["4+ years"])
        provider = _aligned_provider(resume, job).fail_with(ModelLoadError)
        result = _hybrid(resume, job, provider)
        assert result.deterministic.experience_match.score == pytest.approx(25.0)
        assert result.overall_score is not None and result.overall_score > 0
        assert result.overall_score == result.deterministic.overall_score

    def test_no_comparable_content_is_marked_no_content(self) -> None:
        resume = make_resume()
        job = make_job(required=["Python"])
        result = _hybrid(resume, job, _aligned_provider(resume, job))
        assert result.semantic is not None
        assert result.semantic.overall_similarity is None
        assert result.metadata.semantic_availability.status == "no_content"
        assert result.metadata.mode == "deterministic-only"

    def test_no_evidence_yields_none_overall(self) -> None:
        resume = make_resume()
        job = make_job()
        result = _hybrid(resume, job, _aligned_provider(resume, job))
        assert result.overall_score is None
        assert result.metadata.mode == "no-evidence"
        assert result.metadata.evidence_quality == "limited"
        assert result.metadata.semantic_availability.status == "no_content"


class TestHardRequirementSafety:
    def test_missing_required_skill_is_never_a_semantic_match(self) -> None:
        resume = make_resume(skills=["Python", "Docker"])
        job = make_job(required=["Python", "Docker", "Kubernetes"])
        result = _hybrid(resume, job, _aligned_provider(resume, job))

        assert result.missing_required == ["Kubernetes"]
        assert "Kubernetes" not in result.deterministic.skill_match.matched_required
        assert not any("Kubernetes" in entry for entry in result.matched_requirements)
        assert any(
            "Missing required skill: Kubernetes." in gap for gap in result.gaps
        )

        skill_insights = [
            i for i in result.semantic_insights if i.category == "skill"
        ]
        assert skill_insights, "expected a semantic insight for the gap"
        k8s_insights = [i for i in skill_insights if "Kubernetes" in i.statement]
        assert k8s_insights, "expected a Kubernetes-related semantic insight"
        for insight in k8s_insights:
            assert "not explicitly verified" in insight.statement
            assert "does not establish possession" in insight.statement

    def test_exact_match_remains_authoritative(self) -> None:
        resume = make_resume(skills=["Python", "Docker"])
        job = make_job(required=["Python", "Docker"])
        result = _hybrid(resume, job, _aligned_provider(resume, job))
        assert result.missing_required == []
        assert "Required skill matched: Python" in result.matched_requirements
        assert not any(i.category == "skill" for i in result.semantic_insights)

    def test_alias_match_is_explicit_not_semantic(self) -> None:
        resume = make_resume(skills=["K8s"])
        job = make_job(required=["kubernetes"])
        result = _hybrid(resume, job, _aligned_provider(resume, job))
        assert result.missing_required == []
        assert "Required skill matched: kubernetes" in result.matched_requirements
        assert not any(i.category == "skill" for i in result.semantic_insights)

    def test_missing_preferred_skill_is_semantically_noted_not_required(self) -> None:
        resume = make_resume(skills=["Python"])
        job = make_job(required=["Python"], preferred=["Kubernetes"])
        result = _hybrid(resume, job, _aligned_provider(resume, job))
        assert result.missing_required == []
        assert "Kubernetes" not in " ".join(result.deterministic.unmet_requirements)
        assert not any(
            u.startswith("Required skill")
            for u in result.deterministic.unmet_requirements
        )
        skill_insights = [i for i in result.semantic_insights if i.category == "skill"]
        assert any(
            "Preferred skill 'Kubernetes' is not explicitly verified" in i.statement
            for i in skill_insights
        )


class TestSemanticInsights:
    def test_high_relatedness_categories_produce_insights(self) -> None:
        resume = make_resume(
            summary="Backend engineer who ships payments APIs.",
            skills=["Python"],
            projects=[Project(name="Billing", description="payment gateway")],
        )
        job = make_job(
            summary="Build and scale payments APIs.",
            required=["Python"],
            responsibilities=["Operate production payment systems"],
        )
        result = _hybrid(resume, job, _aligned_provider(resume, job))
        categories = {i.category for i in result.semantic_insights}
        assert "project" in categories
        assert any(
            "semantic relatedness" in i.statement and "strong" in i.statement
            for i in result.semantic_insights
        )

    def test_moderate_relatedness_reports_overall_insight(self) -> None:
        resume = make_resume(summary="x", skills=["Python"])
        job = make_job(required=["Python"], summary="y")
        provider = _orthogonal_provider(resume, job)
        result = _hybrid(resume, job, provider)
        assert result.semantic is not None
        assert result.semantic.overall_similarity == pytest.approx(0.5)
        assert any(i.category == "overall" for i in result.semantic_insights)

    def test_opposite_vectors_produce_no_insights(self) -> None:
        resume = make_resume(summary="x", skills=["Python"])
        job = make_job(required=["Python"], summary="y")
        provider = _opposite_provider(resume, job)
        result = _hybrid(resume, job, provider)
        assert result.semantic is not None
        assert result.semantic.overall_similarity == pytest.approx(0.0)
        assert result.semantic_insights == []


class TestPrivacySurface:
    def test_no_persistence(self, tmp_path: Path) -> None:
        resume = make_resume(skills=["Python"])
        job = make_job(required=["Python"])
        before = set(tmp_path.rglob("*")) if tmp_path.exists() else set()
        _hybrid(resume, job, _aligned_provider(resume, job))
        after = set(tmp_path.rglob("*")) if tmp_path.exists() else set()
        assert after == before
