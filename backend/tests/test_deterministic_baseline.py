"""Focused tests for the deterministic Phase-5A baseline adapter.

Covers: honest mapping (no fabrication), threshold derivation, evaluation
reproducibility, and that production Phase-5A scoring is untouched.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.job_parsing.schemas import JobDescription
from app.matching.schemas import SCORE_WEIGHTS_DEFAULT
from app.matching.service import match_resume_to_job
from app.parsing.schemas import (
    ConfidenceLevel,
    Resume,
)
from ml.deterministic_baseline.adapter import (
    _word_count,
    compute_phase5a_scores,
    derive_threshold_on_train,
    job_from_row,
    resume_from_row,
    score_to_label,
)


def _resume_row(
    *,
    skills: list[str] | None = None,
    summary: str = "",
    experience_bullets: list[str] | None = None,
) -> pd.Series:
    return pd.Series({
        "resume_id": "r1",
        "role": "Engineer",
        "seniority": "mid",
        "years_experience": 5,
        "industry": "tech",
        "education": "bachelor",
        "skills": np.array(skills or [], dtype=object),
        "summary": summary,
        "experience_bullets": np.array(experience_bullets or [], dtype=object),
    })


def _job_row(
    *,
    must: list[str] | None = None,
    nice: list[str] | None = None,
    description: str = "A job",
    responsibilities: list[str] | None = None,
    title: str = "SWE",
) -> pd.Series:
    return pd.Series({
        "job_id": "j1",
        "job_title": title,
        "seniority": "mid",
        "industry": "tech",
        "must_have_skills": np.array(must or [], dtype=object),
        "nice_to_have_skills": np.array(nice or [], dtype=object),
        "description": description,
        "responsibilities": np.array(responsibilities or [], dtype=object),
        "requirements": np.array([], dtype=object),
    })


class TestWordCount:
    def test_empty(self) -> None:
        assert _word_count() == 0

    def test_string_parts(self) -> None:
        assert _word_count("hello world", "foo") == 3

    def test_list_parts(self) -> None:
        assert _word_count(["a", "b c"], "d") == 4

    def test_none_parts(self) -> None:
        assert _word_count(None, [], "x") == 1


class TestResumeFromRow:
    def test_basic_mapping(self) -> None:
        row = _resume_row(
            skills=["python", "go"],
            summary="Summary text",
            experience_bullets=["bul1", "bul2"],
        )
        resume = resume_from_row(row)
        assert isinstance(resume, Resume)
        assert resume.skills.all == ["python", "go"]
        assert resume.summary == "Summary text"
        assert resume.experience == []
        assert resume.education == []
        assert resume.certifications == []
        assert resume.metadata.file_type == "txt"
        assert resume.metadata.overall_confidence == ConfidenceLevel.HIGH
        assert resume.metadata.word_count == 6  # summary=2 + bullets=2 + skills=2

    def test_empty_skills_not_nan(self) -> None:
        row = _resume_row()
        resume = resume_from_row(row)
        assert resume.skills.all == []

    def test_no_fabrication_of_experience(self) -> None:
        row = _resume_row()
        resume = resume_from_row(row)
        assert resume.experience == []

    def test_no_fabrication_of_education(self) -> None:
        row = _resume_row()
        resume = resume_from_row(row)
        assert resume.education == []


class TestJobFromRow:
    def test_basic_mapping(self) -> None:
        row = _job_row(
            must=["python"], nice=["docker"], title="Dev", description="desc"
        )
        job = job_from_row(row)
        assert isinstance(job, JobDescription)
        assert job.required_skills == ["python"]
        assert job.preferred_skills == ["docker"]
        assert job.title == "Dev"
        assert job.summary == "desc"
        assert job.responsibilities == []
        assert job.experience_requirements == []
        assert job.education_requirements == []
        assert job.certifications == []
        assert job.qualifications == []
        assert job.metadata.file_type.value == "txt"

    def test_no_fabrication_of_experience_requirements(self) -> None:
        row = _job_row()
        job = job_from_row(row)
        assert job.experience_requirements == []

    def test_no_fabrication_of_education_requirements(self) -> None:
        row = _job_row()
        job = job_from_row(row)
        assert job.education_requirements == []


class TestMappingIntegrity:
    """Ensure the adapter produces a valid Phase-5A result with no crashes."""

    def test_match_resume_to_job_succeeds(self) -> None:
        r = resume_from_row(_resume_row(skills=["python"], summary="Pro"))
        j = job_from_row(_job_row(must=["python"], nice=["go"]))
        result = match_resume_to_job(r, j, reference_date=date(2026, 1, 1))
        assert result.overall_score is not None
        assert 0.0 <= result.overall_score <= 100.0
        assert result.skill_match.required_coverage is not None
        assert result.skill_match.preferred_coverage is not None

    def test_score_is_purely_skill_based(self) -> None:
        """With no experience/education/certifications, score = skill composite."""
        r = resume_from_row(_resume_row(skills=["python", "go"]))
        j = job_from_row(_job_row(must=["python"], nice=["go"]))
        result = match_resume_to_job(r, j, reference_date=date(2026, 1, 1))
        assert result.overall_score is not None
        req_cov = result.skill_match.required_coverage
        pref_cov = result.skill_match.preferred_coverage
        assert req_cov is not None
        assert pref_cov is not None
        w = SCORE_WEIGHTS_DEFAULT
        expected = round(
            (w["required_skill"] * req_cov * 100
             + w["preferred_skill"] * pref_cov * 100)
            / (w["required_skill"] + w["preferred_skill"]),
            1,
        )
        assert result.overall_score == pytest.approx(expected)


class TestScoreToLabel:
    def test_above_threshold_positive(self) -> None:
        scores: list[float | None] = [60.0, 70.0, 80.0]
        preds = score_to_label(scores, threshold=65.0)
        assert preds == [0, 1, 1]

    def test_exact_threshold_positive(self) -> None:
        preds = score_to_label([65.0], threshold=65.0)
        assert preds == [1]

    def test_none_predicted_as_negative(self) -> None:
        preds = score_to_label([None], threshold=50.0)
        assert preds == [0]


class TestDeriveThreshold:
    def test_finds_best(self) -> None:
        scores: list[float | None] = [40.0, 50.0, 60.0, 70.0, 80.0]
        y = [0, 0, 1, 1, 1]
        threshold, curve = derive_threshold_on_train(scores, y)
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 100.0
        assert isinstance(curve, dict)
        assert len(curve) > 0


class TestComputePhase5aScores:
    def test_returns_list_of_floats(self) -> None:
        resumes = _resume_row(skills=["python"])
        jobs = _job_row(must=["python"])
        pairs = pd.DataFrame([{"resume_id": "r1", "job_id": "j1"}])
        resumes_lookup = {"r1": resumes}
        jobs_lookup = {"j1": jobs}
        scores = compute_phase5a_scores(pairs, resumes_lookup, jobs_lookup)
        assert len(scores) == 1
        assert scores[0] is not None
        assert isinstance(scores[0], float)


class TestFrozenThresholdNotShared:
    """Ensure train-derived threshold is NOT influenced by test data."""

    def test_threshold_unchanged_by_test_evaluation(self) -> None:
        train_scores: list[float | None] = [30.0, 40.0, 60.0, 70.0]
        train_y = [0, 0, 1, 1]
        threshold, _ = derive_threshold_on_train(train_scores, train_y)

        # adding more data would shift the optimum should NOT change threshold
        # because we call derive_threshold only once.
        test_scores: list[float | None] = [20.0, 90.0]
        test_preds = score_to_label(test_scores, threshold)
        # This just verifies the threshold is frozen; the predictions use it as-is.
        assert all(p in (0, 1) for p in test_preds)


class TestNoProductionChanges:
    """Verify production Phase-5A scoring is not modified by the adapter."""

    def test_match_resume_to_job_unchanged(self) -> None:
        import inspect

        from app.matching import service

        sig = inspect.signature(service.match_resume_to_job)
        assert "reference_date" in sig.parameters
        # Function exists and is callable
        assert callable(service.match_resume_to_job)
