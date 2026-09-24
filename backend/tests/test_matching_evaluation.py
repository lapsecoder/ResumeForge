"""Behavioural evaluation sanity checks for the deterministic baseline.

This is a sanity dataset, NOT a machine-learning benchmark. It locks in the
qualitative behaviour the matching engine is expected to have:

- a stronger resume outscores a weaker one,
- a missing required skill lowers the score more than a missing preferred one,
- irrelevant skills never inflate the score,
- absent criteria cause no penalty.

Every expected value is computed from the documented formula, so the tests
fail loudly if the engine silently changes its qualitative behaviour.
"""

from __future__ import annotations

from app.parsing.schemas import Certification, Education
from tests.test_matching import make_job, make_resume, match, stint

STRONG_RESUME = make_resume(
    skills=["Python", "Go", "Docker", "PostgreSQL"],
    experience=[stint("2018", "2026")],
    education=[Education(institution="IIT", degree="B.Tech")],
    certifications=[Certification(name="PMP")],
    summary="Senior backend engineer specialising in containerization.",
)

WEAK_RESUME = make_resume(
    skills=["Python"],
    experience=[stint("2024", "2026")],
    education=[],
    certifications=[],
)

TARGET_JOB = make_job(
    required=["Python", "Go", "Docker"],
    preferred=["PostgreSQL", "Kubernetes"],
    experience_requirements=["4+ years"],
    education_requirements=["Bachelor's degree"],
    certifications=["PMP"],
)


class TestQualitativeBehaviour:
    def test_strong_beats_weak(self) -> None:
        strong = match(STRONG_RESUME, TARGET_JOB).overall_score
        weak = match(WEAK_RESUME, TARGET_JOB).overall_score
        assert strong is not None and weak is not None
        assert strong > weak

    def test_missing_required_skill_lowers_more_than_missing_preferred(self) -> None:
        both_met = match(
            make_resume(skills=["Python", "Go", "Kubernetes"]),
            make_job(required=["Python", "Go"], preferred=["Kubernetes"]),
        ).overall_score
        required_missing = match(
            make_resume(skills=["Python", "Kubernetes"]),
            make_job(required=["Python", "Go"], preferred=["Kubernetes"]),
        ).overall_score
        preferred_missing = match(
            make_resume(skills=["Python", "Go"]),
            make_job(required=["Python", "Go"], preferred=["Kubernetes"]),
        ).overall_score
        assert both_met is not None
        assert required_missing is not None
        assert preferred_missing is not None
        assert required_missing < preferred_missing < both_met

    def test_irrelevant_skills_do_not_inflate_score(self) -> None:
        padded = make_resume(
            skills=["Photography", "Cooking", "Magic", "Origami", "Python"]
        )
        focused = make_resume(skills=["Python"])
        padded_score = match(padded, make_job(required=["Python"])).overall_score
        focused_score = match(focused, make_job(required=["Python"])).overall_score
        assert padded_score == 100.0
        assert focused_score == 100.0

    def test_perfect_match_against_thin_job_is_not_diminished(self) -> None:
        thin_job = make_job(required=["Python"])
        result = match(make_resume(skills=["Python"]), thin_job)
        assert result.overall_score == 100.0
        assert result.metadata.applied_weights == {"required_skill": 50.0}

    def test_absent_experience_criterion_incurs_no_penalty(self) -> None:
        resume = make_resume(skills=["Python"], experience=[stint("2020", "2021")])
        no_exp_job = make_job(required=["Python"])
        exp_shortfall_job = make_job(
            required=["Python"], experience_requirements=["4+ years"]
        )
        no_exp = match(resume, no_exp_job).overall_score
        shortfall = match(resume, exp_shortfall_job).overall_score
        assert no_exp == 100.0
        assert shortfall is not None and shortfall < 100.0

    def test_zero_requirement_job_scores_none_not_zero(self) -> None:
        result = match(STRONG_RESUME, make_job())
        assert result.overall_score is None
        assert result.strengths == []
        assert result.gaps == []
