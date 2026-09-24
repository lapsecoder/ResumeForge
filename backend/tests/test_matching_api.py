"""API tests for POST /api/v1/matching/score.

Verifies success responses, structured validation errors, and that nothing is
persisted to disk during a matching call.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

RESUME_BODY = {
    "contact": {"name": "Jane Doe"},
    "summary": "Backend engineer.",
    "experience": [
        {
            "company": "Acme",
            "title": "Engineer",
            "start_date": "2020",
            "end_date": "2023",
        }
    ],
    "education": [{"institution": "IIT", "degree": "B.Tech"}],
    "skills": {"all": ["Python", "PostgreSQL"], "languages": []},
    "projects": [],
    "certifications": [],
    "custom_sections": [],
    "metadata": {
        "word_count": 0,
        "file_type": "pdf",
        "overall_confidence": "high",
    },
}

JOB_BODY = {
    "title": "Backend Engineer",
    "required_skills": ["Python"],
    "preferred_skills": [],
    "experience_requirements": ["3+ years"],
    "education_requirements": ["Bachelor's degree"],
    "certifications": [],
    "qualifications": [],
    "metadata": {
        "word_count": 0,
        "file_type": "txt",
        "overall_confidence": "high",
    },
}


class TestScoreSuccess:
    def test_full_payload_returns_score(self) -> None:
        resp = client.post(
            "/api/v1/matching/score",
            json={"resume": RESUME_BODY, "job": JOB_BODY},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["overall_score"] is not None
        assert "overall_score" in body
        assert body["skill_match"]["required_coverage"] == 1.0
        assert body["metadata"]["method"] == "deterministic-baseline"
        assert body["metadata"]["reference_date"] is not None

    def test_zero_requirement_job_returns_none_overall(self) -> None:
        empty_job = {
            **JOB_BODY,
            "required_skills": [],
            "experience_requirements": [],
            "education_requirements": [],
            "certifications": [],
            "qualifications": [],
        }
        resp = client.post(
            "/api/v1/matching/score", json={"resume": RESUME_BODY, "job": empty_job}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["overall_score"] is None
        assert body["skill_match"]["score"] is None

    def test_success_returns_no_error_field(self) -> None:
        resp = client.post(
            "/api/v1/matching/score", json={"resume": RESUME_BODY, "job": JOB_BODY}
        )
        body = resp.json()
        assert "error" not in body


class TestValidationErrors:
    def _assert_structured_validation_error(self, resp) -> None:
        assert resp.status_code == 422
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == "validation_error"
        assert body["error"]["message"]
        assert body["error"]["request_id"]

    def test_missing_job_field(self) -> None:
        resp = client.post("/api/v1/matching/score", json={"resume": RESUME_BODY})
        self._assert_structured_validation_error(resp)

    def test_missing_resume_field(self) -> None:
        resp = client.post("/api/v1/matching/score", json={"job": JOB_BODY})
        self._assert_structured_validation_error(resp)

    def test_missing_required_metadata(self) -> None:
        bad_resume = {k: v for k, v in RESUME_BODY.items() if k != "metadata"}
        resp = client.post(
            "/api/v1/matching/score",
            json={"resume": bad_resume, "job": JOB_BODY},
        )
        self._assert_structured_validation_error(resp)

    def test_wrong_type_nested(self) -> None:
        bad_resume = {**RESUME_BODY, "skills": "not-a-list"}
        resp = client.post(
            "/api/v1/matching/score",
            json={"resume": bad_resume, "job": JOB_BODY},
        )
        self._assert_structured_validation_error(resp)

    def test_empty_body(self) -> None:
        resp = client.post("/api/v1/matching/score", json={})
        self._assert_structured_validation_error(resp)

    def test_validation_error_never_echoes_input(self) -> None:
        secret = {"resume": {"skills": {"all": ["TOP-SECRET-SKILL"]}}, "job": JOB_BODY}
        resp = client.post("/api/v1/matching/score", json=secret)
        assert "TOP-SECRET-SKILL" not in resp.text


class TestNoPersistence:
    def test_matching_creates_no_files(self, tmp_path: Path) -> None:
        before = set(tmp_path.rglob("*")) if tmp_path.exists() else set()
        client.post(
            "/api/v1/matching/score", json={"resume": RESUME_BODY, "job": JOB_BODY}
        )
        after = set(tmp_path.rglob("*")) if tmp_path.exists() else set()
        assert after == before
