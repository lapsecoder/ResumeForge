"""Synthetic-evaluation tests for the hybrid matching engine (Phase 5C).

Seven deliberately controlled scenarios exercise the intended behaviour of the
hybrid composer: exact-over-semantic authority, graceful downgrade, no
inflation from irrelevant or padded content, and honest labelling of missing
or unverifiable criteria. All similarities are produced by fake providers, so
the suite runs offline and deterministically.
"""

from __future__ import annotations

import pytest

from app.hybrid_matching.service import compute_hybrid_match
from app.job_parsing.schemas import JobDescription, JobMetadata
from app.parsing.schemas import ConfidenceLevel, Education, Project, Resume
from app.semantic_matching.model import ModelMetadata
from app.semantic_matching.text_builder import build_job_units, build_resume_units
from tests.test_matching import REFERENCE, make_job, make_resume, stint

IOTA = [1.0, 0.0]
ORTHOGONAL = [0.0, 1.0]


class VectorProvider:
    def __init__(self, vectors: dict[str, list[float]], default: list[float] = IOTA):
        self._vectors = vectors
        self._default = default

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors.get(text, self._default) for text in texts]

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(model_name="eval", model_dimension=2, device="cpu")


def _build(resume: Resume, job: JobDescription) -> dict[str, list[float]]:
    mapping: dict[str, list[float]] = {}
    for unit in build_resume_units(resume):
        mapping[unit.text] = list(IOTA)
    for unit in build_job_units(job):
        mapping[unit.text] = list(IOTA)
    return mapping


def _provider(
    resume: Resume, job: JobDescription, *, semantic_weak: bool = False
) -> VectorProvider:
    mapping = _build(resume, job)
    if semantic_weak:
        for unit in build_job_units(job):
            mapping[unit.text] = list(ORTHOGONAL)
    return VectorProvider(mapping)


def _hybrid(resume: Resume, job: JobDescription, *, semantic_weak: bool = False):
    return compute_hybrid_match(
        resume,
        job,
        reference_date=REFERENCE,
        provider=_provider(resume, job, semantic_weak=semantic_weak),
    )


def _edu(school: str, degree: str) -> Education:
    return Education(institution=school, degree=degree)


def _project(text: str) -> Project:
    return Project(name="relevant", description=text)


class TestScenarioSet:
    def test_s1_strong_exact_and_strong_semantic(self) -> None:
        """A clear, fully-matching candidate scores full marks."""
        resume = make_resume(
            skills=["Python", "PostgreSQL", "Docker"],
            experience=[stint("2019", "2026")],
            education=[_edu("MIT", "B.Tech")],
        )
        job = make_job(
            required=["Python", "PostgreSQL"],
            preferred=["Docker"],
            experience_requirements=["5+ years"],
            education_requirements=["Bachelor's degree"],
        )
        result = _hybrid(resume, job)
        assert result.overall_score == 100.0
        assert result.metadata.mode == "hybrid"
        assert result.metadata.evidence_quality == "high"
        assert result.missing_required == []
        assert result.semantic is not None
        assert result.semantic.overall_similarity == 1.0

    def test_s2_strong_exact_but_weak_semantic(self) -> None:
        """The hybrid stays high but reflects the weaker semantic evidence."""
        resume = make_resume(
            skills=["Python", "PostgreSQL"],
            experience=[stint("2019", "2026")],
            education=[_edu("MIT", "B.Tech")],
        )
        job = make_job(
            required=["Python", "PostgreSQL"],
            experience_requirements=["5+ years"],
            education_requirements=["Bachelor's degree"],
        )
        result = _hybrid(resume, job, semantic_weak=True)
        assert result.deterministic.overall_score == 100.0
        assert result.overall_score == pytest.approx(85.0)  # 0.70*100 + 0.30*50
        assert result.metadata.mode == "hybrid"
        assert any(i.category == "overall" for i in result.semantic_insights)

    def test_s3_weak_exact_but_strong_semantic_relatedness(self) -> None:
        """Relatedness can nudge the composite but never claims possession."""
        resume = make_resume(
            skills=["Python"],
            projects=[_project("Deploy ML services on Kubernetes clusters")],
        )
        job = JobDescription(
            required_skills=["Kubernetes"],
            responsibilities=["Orchestrate Kubernetes clusters at scale."],
            metadata=JobMetadata(word_count=0, overall_confidence=ConfidenceLevel.HIGH),
        )
        result = _hybrid(resume, job)
        assert result.deterministic.overall_score == 0.0
        assert result.overall_score == pytest.approx(30.0)  # 0.30 * (semantic 1.0)
        assert result.missing_required == ["Kubernetes"]
        skill_insights = [i for i in result.semantic_insights if i.category == "skill"]
        assert any(
            "does not establish possession" in i.statement for i in skill_insights
        )

    def test_s4_missing_required_skill_never_becomes_a_match(self) -> None:
        """A strong semantic overlap cannot flip a hard requirement."""
        resume = make_resume(skills=["Python", "Docker"])
        job = make_job(required=["Python", "Docker", "Kubernetes"])
        result = _hybrid(resume, job)
        assert result.overall_score == pytest.approx(76.7)  # 0.70*(66.7) + 30
        assert result.missing_required == ["Kubernetes"]
        assert "Kubernetes" not in result.deterministic.skill_match.matched_required
        assert not any("Kubernetes" in g for g in result.matched_requirements)

    def test_s5_irrelevant_padding_inflates_nothing(self) -> None:
        """Extra, unrequested resume content must not raise the score."""
        focused = make_resume(skills=["Python"])
        padded = make_resume(
            skills=["Python", "Photography", "Cooking", "Chess tournaments"]
        )
        job = make_job(required=["Python"])
        focused_result = _hybrid(focused, job)
        padded_result = _hybrid(padded, job)
        assert focused_result.overall_score == 100.0
        assert padded_result.overall_score == focused_result.overall_score
        assert padded_result.missing_required == []

    def test_s6_missing_optional_criteria_is_not_a_penalty(self) -> None:
        """Preferred-skill gaps are reported honestly, not as requirements."""
        resume = make_resume(skills=["Python"])
        job = make_job(required=["Python"], preferred=["Kubernetes"])
        result = _hybrid(resume, job)
        assert result.overall_score == pytest.approx(83.8)  # 0.70*76.9 + 30
        assert result.missing_required == []
        assert "Kubernetes" not in " ".join(result.deterministic.unmet_requirements)
        assert "Kubernetes" in " ".join(
            result.deterministic.skill_match.missing_preferred
        )
        assert any(
            "Preferred skill 'Kubernetes' is not explicitly verified" in i.statement
            for i in result.semantic_insights
        )

    def test_s7_missing_resume_sections_are_insufficient_not_absence(self) -> None:
        """When resume sections are missing, the scorer says 'not verifiable'."""
        resume = make_resume(skills=["Python"])
        job = make_job(required=["Python"], experience_requirements=["4+ years"])
        result = _hybrid(resume, job)
        assert result.overall_score == 100.0
        assert result.deterministic.experience_match.score is None
        assert any(
            "could not be verified" in gap
            or "insufficient data" in gap
            or "not verifiable" in gap
            for gap in result.gaps
        )
        assert any(
            "not treated as zero experience" in gap for gap in result.gaps
        ), "absence of experience dates must not be scored as zero experience"
