"""Privacy guarantees of the ML audit pipeline.

Audit outputs are aggregates only: raw markers (a stand-in for PII and
resume/job text) inserted into input frames must never surface in any
function result, report markdown, or rendered artefact.
"""

from __future__ import annotations

import json

import pandas as pd

from ml.audit.leakage import leak_resumes
from ml.audit.quality import quality_diagnose
from ml.audit.report import markdown_report
from ml.audit.target import derivability, resume_exposure

_MARKER = "SECRET-MARKER-6C"
_EMAIL = "private@example.com"


def _frames():
    resumes = pd.DataFrame(
        {
            "resume_id": ["R_1", "R_2"],
            "role": [_MARKER, "Engineer"],
            "seniority": ["Mid", "Senior"],
            "years_experience": [4, 8],
            "industry": ["Tech", "Finance"],
            "education": ["B.Tech", "M.Sc"],
            "skills": [["Python", "Docker"], ["Python"]],
            "summary": [f"contact {_EMAIL} note {_MARKER}", "plain summary"],
            "experience_bullets": [["bulleted"], ["bulleted"]],
        }
    )
    jobs = pd.DataFrame(
        {
            "job_id": ["J_1", "J_2"],
            "job_title": ["Backend Engineer", "Engineer"],
            "seniority": ["Mid", "Senior"],
            "industry": ["Tech", "Finance"],
            "must_have_skills": [["Python", "Docker"], ["Python"]],
            "nice_to_have_skills": [[], []],
            "description": [f"role {_MARKER}", "x"],
            "responsibilities": [["build APIs"], ["go"]],
            "requirements": [[], []],
        }
    )
    matches = pd.DataFrame(
        {
            "job_id": ["J_1", "J_2"],
            "relevant_resume_ids": [["R_1"], ["R_2"]],
        }
    )
    return resumes, jobs, matches


def _serialized_text(value) -> str:
    return json.dumps(value, sort_keys=True)


class TestAggregatesNeverEchoInput:
    def test_leakage_output_has_no_marker(self) -> None:
        resumes, _, _ = _frames()
        assert _MARKER not in _serialized_text(leak_resumes(resumes))

    def test_quality_output_exposes_neither_marker_nor_email(self) -> None:
        resumes, jobs, matches = _frames()
        result = quality_diagnose(resumes, jobs, matches)
        assert _MARKER not in _serialized_text(result)
        assert _EMAIL not in _serialized_text(result)

    def test_derivability_output_has_no_resume_ids(self) -> None:
        resumes, jobs, matches = _frames()
        result = derivability(resumes, jobs, matches)
        assert "R_1" not in _serialized_text(result)

    def test_exposure_output_has_no_resume_ids(self) -> None:
        _, _, matches = _frames()
        assert "R_1" not in _serialized_text(resume_exposure(matches))

    def test_markdown_report_has_no_marker(self) -> None:
        payload = {
            "dataset": {"id": "x", "revision": "r", "frame_counts": {}},
            "statistics": {
                "resumes": {
                    "count": 2,
                    "columns": [],
                    "seniority_counts": {"Mid": 1},
                    "years_experience": {"mean": 6.0, "min": 4, "max": 8},
                    "skills_per_resume": {"mean": 1.5},
                    "top_20_skills": [["Python", 2]],
                },
                "jobs": {
                    "count": 2,
                    "columns": [],
                    "seniority_counts": {},
                    "must_have_per_job": {"mean": 1.5},
                    "top_20_must_have": [["Python", 1]],
                },
            },
            "target": {
                "rows": 2, "unique_jobs": 2,
                "per_job_relevant_count": {"distinct_values": [1]},
            },
            "exposure": {"distinct_resumes_used": 2, "histogram_buckets": {"1-2": 2}},
            "derivability": {
                "threshold_must_have": 0.6,
                "cap": 30,
                "published_relevant_pairs": 2,
                "rule_candidate_pairs": 3,
                "pct_relevant_published_within_candidate": 100.0,
                "fraction_candidate_pairs_discarded_by_cap": 0.0,
                "jobs_with_more_candidates_than_cap": 0,
            },
            "leakage": {
                "resumes": {
                    "pct_summary_follows_template": 100.0,
                    "experience_bullets": {
                        "pct_occurrences_of_shared_strings": 100.0,
                        "distinct_strings": 1,
                    },
                },
                "jobs": {},
                "vocab": {
                    "shared_narrative_words_fraction": 49.5,
                    "skills_shared_fraction": 100.0,
                },
            },
            "quality": {"resumes": {}, "jobs": {}, "matches": {}},
            "split_strategy": {
                "has_repository_test_split": False,
                "recommendations": ["a", "b", "c"],
            },
            "baselines": {"A": "desc"},
        }
        report = markdown_report(payload)
        assert _MARKER not in report
        assert _EMAIL not in report
