"""Offline behavioral tests for the ML dataset audit modules.

Uses tiny in-memory frames; no disk, no network, no real dataset.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml.audit.leakage import leak_resumes, vocab_overlap
from ml.audit.quality import quality_diagnose
from ml.audit.splits import split_strategy_recommendation
from ml.audit.statistics import summarize_jobs, summarize_resumes
from ml.audit.target import derivability, resume_exposure, summarize_matches


def _frames():
    resumes = pd.DataFrame(
        {
            "resume_id": ["R_1", "R_2"],
            "role": ["Backend Engineer", "Data Scientist"],
            "seniority": ["Mid", "Senior"],
            "years_experience": [4, 8],
            "industry": ["Tech", "Finance"],
            "education": ["B.Tech", "M.Sc"],
            "skills": [["Python", "Docker"], ["Python"]],
            "summary": [
                "Backend Engineer with 4 years of experience in Tech",
                "Data Scientist with 8 years of experience in Finance",
            ],
            "experience_bullets": [["shipped API"], ["shipped API"]],
        }
    )
    jobs = pd.DataFrame(
        {
            "job_id": ["J_1", "J_2"],
            "job_title": ["Backend Engineer", "Data Scientist"],
            "seniority": ["Mid", "Senior"],
            "industry": ["Tech", "Finance"],
            "must_have_skills": [["Python", "Docker"], ["Python"]],
            "nice_to_have_skills": [["Kubernetes"], []],
            "description": ["desc1", "desc2"],
            "responsibilities": [["build APIs"], ["analyze data"]],
            "requirements": [[], []],
        }
    )
    matches = pd.DataFrame(
        {
            "job_id": ["J_1", "J_2"],
            "relevant_resume_ids": [["R_1", "R_2"], ["R_2"]],
        }
    )
    return resumes, jobs, matches


class TestStatistics:
    def test_resume_summary(self) -> None:
        resumes, _, _ = _frames()
        stats = summarize_resumes(resumes)
        assert stats["count"] == 2
        assert stats["nulls_non_optional"]["resume_id"] == 0
        assert stats["years_experience"]["mean"] == 6.0
        assert stats["seniority_counts"] == {"Mid": 1, "Senior": 1}

    def test_job_summary(self) -> None:
        _, jobs, _ = _frames()
        stats = summarize_jobs(jobs)
        assert stats["count"] == 2
        assert stats["must_have_per_job"]["mean"] == 1.5

    def test_top_skills_resumes(self) -> None:
        resumes, _, _ = _frames()
        top = dict(summarize_resumes(resumes)["top_20_skills"])
        assert top["Python"] == 2
        assert top["Docker"] == 1


class TestTarget:
    def test_derivability_reachability(self) -> None:
        resumes, jobs, matches = _frames()
        result = derivability(resumes, jobs, matches)
        # J_1 must = [Python, Docker]; only R_1 is a candidate (R_2 lacks
        # Docker). J_2 must = [Python]; both are candidates.
        # overlap with published (3 pairs) = R_1(J_1) + R_2(J_2) = 2 -> 66.7%.
        assert (
            result["pct_relevant_published_within_candidate"]
            == pytest.approx(200.0 / 3.0)
        )
        assert result["published_relevant_pairs"] == 3
        assert result["rule_candidate_pairs"] == 3

    def test_matches_lists_are_30_free(self) -> None:
        _, _, matches = _frames()
        summarized = summarize_matches(matches)
        assert summarized["per_job_relevant_count"]["distinct_values"] == [1, 2]

    def test_exposure_counts(self) -> None:
        _, _, matches = _frames()
        exposure = resume_exposure(matches)
        assert exposure["distinct_resumes_used"] == 2
        assert exposure["counts_per_resume"]["max"] == 2


class TestLeakage:
    def test_summary_template_detected(self) -> None:
        resumes, _, _ = _frames()
        assert leak_resumes(resumes)["pct_summary_follows_template"] == 100.0

    def test_shared_bullet_strings(self) -> None:
        resumes, _, _ = _frames()
        bullets = leak_resumes(resumes)["experience_bullets"]
        assert bullets["distinct_strings"] == 1
        assert bullets["pct_occurrences_of_shared_strings"] == 100.0

    def test_shared_vocab(self) -> None:
        resumes, jobs, _ = _frames()
        overlap = vocab_overlap(resumes, jobs)
        assert overlap["skills_shared_fraction"] == 100.0


class TestQuality:
    def test_email_detected_in_resume(self) -> None:
        resumes, _, _ = _frames()
        resumes.loc[0, "summary"] = "contact test@example.com"
        result = quality_diagnose(resumes, _frames()[1], _frames()[2])
        assert result["resumes"]["detected_email_like"] >= 1

    def test_missing_job_reference(self) -> None:
        resumes, jobs, matches = _frames()
        matches.loc[0, "job_id"] = "J_MISSING"
        result = quality_diagnose(resumes, jobs, matches)
        assert result["matches"]["rows_referencing_missing_job"] == 1

    def test_empty_flags(self) -> None:
        resumes, jobs, _ = _frames()
        result = quality_diagnose(resumes, jobs, _frames()[2])
        assert result["resumes"]["empty_summary"] == 0
        assert result["jobs"]["empty_must_have"] == 0


class TestSplits:
    def test_reports_no_repository_split(self) -> None:
        resumes, jobs, matches = _frames()
        recommendation = split_strategy_recommendation(resumes, jobs, matches)
        assert recommendation["has_repository_test_split"] is False
        assert len(recommendation["recommendations"]) >= 3
