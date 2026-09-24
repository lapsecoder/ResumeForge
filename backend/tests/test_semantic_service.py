"""Semantic service tests with deterministic, hand-controlled vectors.

The fake provider returns vectors keyed by exact unit text, so every similarity
value is predictable from cosine math alone. No model library is required.
"""

from __future__ import annotations

import pytest

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
from app.semantic_matching.model import InferenceError, ModelMetadata
from app.semantic_matching.service import compute_semantic_match
from app.semantic_matching.text_builder import build_job_units, build_resume_units


class FakeEmbeddingProvider:
    """Returns vectors keyed by exact text; defaults to [0.0, 0.0]."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self.prompts: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.prompts.extend(texts)
        default = next(iter(self._vectors.values()), [0.0, 0.0])
        return [self._vectors.get(text, default) for text in texts]

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(model_name="fake-model", model_dimension=2, device="cpu")


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
    title: str | None = None,
    summary: str | None = None,
    required: list[str] | None = None,
    responsibilities: list[str] | None = None,
    experience: list[str] | None = None,
    education: list[str] | None = None,
    certifications: list[str] | None = None,
    qualifications: list[str] | None = None,
) -> JobDescription:
    return JobDescription(
        title=title,
        summary=summary,
        required_skills=required or [],
        responsibilities=responsibilities or [],
        experience_requirements=experience or [],
        education_requirements=education or [],
        certifications=certifications or [],
        qualifications=qualifications or [],
        metadata=_job_metadata(),
    )


def provider_for(
    resume: Resume, job: JobDescription, **vectors
) -> FakeEmbeddingProvider:
    """Build a provider where every unit text gets its explicit vector."""
    texts = [unit.text for unit in build_resume_units(resume) + build_job_units(job)]
    mapping = {text: list(vectors.get(text, [1.0, 0.0])) for text in texts}
    return FakeEmbeddingProvider(mapping)


def test_exact_skill_similarity_is_one() -> None:
    resume = make_resume(skills=["Python"])
    job = make_job(required=["Python"])
    result = compute_semantic_match(resume, job, provider=provider_for(resume, job))
    assert result.skill_similarity == pytest.approx(1.0)
    assert result.overall_similarity == pytest.approx(1.0)
    assert result.overall_cosine == pytest.approx(1.0)
    assert {item.job_unit_name for item in result.matched_semantic_items} == {
        "required-skill:0"
    }
    assert result.matched_semantic_items[0].resume_unit_name == "skill:0"
    assert result.metadata.model_name == "fake-model"
    assert result.metadata.device == "cpu"


def test_relatedness_is_not_deterministic_possession() -> None:
    resume = make_resume(projects=[Project(name="NLP", description="language models")])
    job = make_job(responsibilities=["fine-tune LLMs"])
    result = compute_semantic_match(resume, job, provider=provider_for(resume, job))
    assert result.project_similarity == pytest.approx(1.0)
    assert {item.job_unit_name for item in result.matched_semantic_items} == {
        "responsibility:0"
    }
    assert "not a hiring probability" in result.note


def test_orthogonal_skill_scores_moderate() -> None:
    resume = make_resume(skills=["Python"])
    job = make_job(required=["Unrelated-Orthogonal"])
    provider = provider_for(resume, job)
    texts = [unit.text for unit in build_resume_units(resume) + build_job_units(job)]
    provider._vectors[texts[0]] = [1.0, 0.0]
    provider._vectors[texts[1]] = [0.0, 1.0]
    result = compute_semantic_match(resume, job, provider=provider)
    assert result.skill_similarity == pytest.approx(0.5)
    assert result.related_items[0].job_unit_name == "required-skill:0"
    assert result.related_items[0].cosine == pytest.approx(0.0)


def test_opposite_skill_scores_low() -> None:
    resume = make_resume(skills=["Python"])
    job = make_job(required=["Bin-Applied-Stats"])
    provider = provider_for(resume, job)
    texts = [unit.text for unit in build_resume_units(resume) + build_job_units(job)]
    provider._vectors[texts[0]] = [1.0, 0.0]
    provider._vectors[texts[1]] = [-1.0, 0.0]
    result = compute_semantic_match(resume, job, provider=provider)
    assert result.skill_similarity == pytest.approx(0.0)
    assert result.low_similarity_items[0].job_unit_name == "required-skill:0"
    assert result.matched_semantic_items == []


class TestMissingSections:
    def test_missing_experience_does_not_penalize_overall(self) -> None:
        resume = make_resume(skills=["Python"])
        job = make_job(required=["Python"], responsibilities=["Ship payments"])
        result = compute_semantic_match(resume, job, provider=provider_for(resume, job))
        assert result.skill_similarity == pytest.approx(1.0)
        assert result.responsibility_similarity is None
        assert result.overall_similarity == pytest.approx(1.0)

    def test_empty_resume_returns_informative_result(self) -> None:
        resume = make_resume()
        job = make_job(required=["Python"])
        result = compute_semantic_match(resume, job, provider=provider_for(resume, job))
        assert result.overall_similarity is None
        assert result.matched_semantic_items == []
        assert "comparable content" in result.note
        assert result.metadata.device == "not-loaded"

    def test_empty_job_returns_informative_result(self) -> None:
        resume = make_resume(skills=["Python"])
        job = make_job()
        result = compute_semantic_match(resume, job, provider=provider_for(resume, job))
        assert result.overall_similarity is None
        assert "comparable content" in result.note


class TestAllCategories:
    def test_every_present_category_scores_one(self) -> None:
        resume = make_resume(
            summary="Backend engineer",
            skills=["Python"],
            experience=[WorkExperience(company="Acme", title="Engineer")],
            projects=[Project(name="Dashboard")],
            education=[Education(institution="IIT", degree="B.Tech")],
            certifications=[Certification(name="AWS")],
        )
        job = make_job(
            title="Backend Engineer",
            summary="Build backend systems",
            required=["Python"],
            responsibilities=["Ship APIs"],
            experience=["3+ years"],
            education=["Bachelor's degree"],
            qualifications=["CS degree"],
            certifications=["AWS"],
        )
        result = compute_semantic_match(resume, job, provider=provider_for(resume, job))
        assert result.summary_similarity == pytest.approx(1.0)
        assert result.skill_similarity == pytest.approx(1.0)
        assert result.experience_similarity == pytest.approx(1.0)
        assert result.responsibility_similarity == pytest.approx(1.0)
        assert result.project_similarity == pytest.approx(1.0)
        assert result.qualification_similarity == pytest.approx(1.0)
        assert result.overall_similarity == pytest.approx(1.0)

    def test_overall_averages_only_present_categories(self) -> None:
        resume = make_resume(summary="Backend engineer", skills=["Python"])
        job = make_job(summary="Backend engineer", required=["Orthogonal"])
        provider = provider_for(resume, job)
        texts = [
            unit.text
            for unit in build_resume_units(resume) + build_job_units(job)
        ]
        # summary texts match; the skill text is orthogonal to the requirement
        provider._vectors[texts[0]] = [1.0, 0.0]
        provider._vectors[texts[1]] = [1.0, 0.0]
        provider._vectors[texts[2]] = [1.0, 0.0]
        provider._vectors[texts[3]] = [0.0, 1.0]
        result = compute_semantic_match(resume, job, provider=provider)
        assert result.summary_similarity == pytest.approx(1.0)
        assert result.skill_similarity == pytest.approx(0.5)
        assert result.overall_similarity == pytest.approx(0.75)

    def test_qualification_pool_spans_education_and_certifications(self) -> None:
        resume = make_resume(
            education=[Education(institution="IIT", degree="B.Tech")],
            certifications=[Certification(name="PMP")],
        )
        job = make_job(
            education=["Bachelor's degree"],
            certifications=["PMP"],
            qualifications=["Analytical thinker"],
        )
        result = compute_semantic_match(resume, job, provider=provider_for(resume, job))
        assert result.qualification_similarity == pytest.approx(1.0)
        assert result.overall_similarity == pytest.approx(1.0)


class TestRefusalToInvent:
    def test_no_matches_when_job_has_qualification_and_resume_has_none(self) -> None:
        resume = make_resume(skills=["Python"], projects=[Project(name="D")])
        job = make_job(certifications=["CISSP"])
        result = compute_semantic_match(resume, job, provider=provider_for(resume, job))
        assert result.qualification_similarity is None
        assert not result.matched_semantic_items
        assert not any("CISSP" in item.job_unit_name for item in result.related_items)

    def test_unrelated_content_stays_out_of_matched_items(self) -> None:
        resume = make_resume(
            skills=["Python"],
            experience=[WorkExperience(company="Acme", title="Eng")],
        )
        job = make_job(
            required=["Python"], responsibilities=["Maintain the on-call rotation"]
        )
        # experience and responsibility texts orthogonal -> moderate, not matched
        provider = provider_for(resume, job)
        texts = [
            unit.text
            for unit in build_resume_units(resume) + build_job_units(job)
        ]
        provider._vectors[texts[0]] = [1.0, 0.0]  # skill
        provider._vectors[texts[1]] = [1.0, 0.0]  # experience
        provider._vectors[texts[2]] = [0.0, 1.0]  # responsibility orthogonal
        provider._vectors[texts[3]] = [1.0, 0.0]  # required skill
        result = compute_semantic_match(resume, job, provider=provider)
        assert result.responsibility_similarity == pytest.approx(0.5)
        assert {item.job_unit_name for item in result.matched_semantic_items} == {
            "required-skill:0"
        }
        assert {item.job_unit_name for item in result.related_items} == {
            "responsibility:0"
        }


class TestErrorPaths:
    def test_dimension_mismatch_raises_inference_error(self) -> None:
        resume = make_resume(skills=["Python"], summary="x")
        job = make_job(required=["Python"], summary="y")
        provider = provider_for(resume, job)
        texts = [
            unit.text
            for unit in build_resume_units(resume) + build_job_units(job)
        ]
        provider._vectors[texts[0]] = [1.0, 0.0]
        provider._vectors[texts[1]] = [1.0, 0.0, 0.0]  # different dimension
        with pytest.raises(InferenceError):
            compute_semantic_match(resume, job, provider=provider)

    def test_provider_inference_error_propagates(self) -> None:
        class BrokenProvider(FakeEmbeddingProvider):
            def embed_texts(self, texts):
                raise InferenceError("mock failure")

        resume = make_resume(skills=["Python"])
        job = make_job(required=["Python"])
        with pytest.raises(InferenceError):
            compute_semantic_match(resume, job, provider=BrokenProvider({}))
