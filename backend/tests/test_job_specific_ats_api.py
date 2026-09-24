"""API tests for the Job-Specific ATS Coverage endpoint.

End-to-end with synthetic resume + job-description JSON. All processing is
transient: no database, no files, no external services.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_RESUME = {
    "contact": {"name": "Jane Doe", "email": "jane@example.com"},
    "summary": "Software engineer using Python and Docker daily.",
    "experience": [
        {
            "company": "Acme",
            "title": "Engineer",
            "description": "Backend work.",
            "achievements": [
                "Built Python services in production.",
                "Deployed containers with Docker.",
            ],
            "skills_mentioned": ["Python", "Docker"],
        }
    ],
    "education": [
        {"institution": "IIT", "degree": "B.Tech", "field": "Computer Science"}
    ],
    "skills": {
        "technical": ["Python", "FastAPI", "Docker", "PostgreSQL"],
        "tools": ["Git"],
        "languages": ["English"],
        "soft": [],
        "all": [],
    },
    "projects": [
        {"name": "P", "description": "Built an API.", "technologies": ["FastAPI"]}
    ],
    "certifications": [],
    "custom_sections": [],
    "metadata": {
        "word_count": 200,
        "file_type": "pdf",
        "overall_confidence": "high",
        "section_confidence": [],
        "section_order": [],
    },
}

_JOB = {
    "required_skills": ["Python", "Docker", "Kafka"],
    "preferred_skills": ["Kubernetes"],
    "responsibilities": ["Build and run data pipelines."],
    "qualifications": [],
    "experience_requirements": [],
    "education_requirements": [],
    "certifications": [],
    "nice_to_have": [],
    "metadata": {
        "word_count": 0, "file_type": "txt",
        "overall_confidence": "high", "section_confidence": [],
    },
}


class TestJobSpecificEndpointSuccess:
    def test_returns_200_and_shape(self) -> None:
        resp = client.post(
            "/api/v1/resumes/job-specific-ats",
            json={"resume": _RESUME, "job_description": _JOB},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "overall_score" in body
        assert "score_label" in body
        assert "category_scores" in body
        assert "coverage" in body
        assert "term_matches" in body
        assert "findings" in body
        assert "metadata" in body

    def test_score_bounded(self) -> None:
        resp = client.post(
            "/api/v1/resumes/job-specific-ats",
            json={"resume": _RESUME, "job_description": _JOB},
        )
        body = resp.json()
        assert 0.0 <= body["overall_score"] <= 100.0
        for cs in body["category_scores"]:
            if cs["score"] is not None:
                assert 0.0 <= cs["score"] <= 100.0

    def test_method_name_is_distinct_from_phase_6a(self) -> None:
        resp = client.post(
            "/api/v1/resumes/job-specific-ats",
            json={"resume": _RESUME, "job_description": _JOB},
        )
        assert (
            resp.json()["metadata"]["method"]
            == "job-specific-ats-coverage-heuristic"
        )

    def test_deterministic_repeat(self) -> None:
        payload = {"resume": _RESUME, "job_description": _JOB}
        r1 = client.post("/api/v1/resumes/job-specific-ats", json=payload)
        r2 = client.post("/api/v1/resumes/job-specific-ats", json=payload)
        assert r1.json() == r2.json()

    def test_coverage_fractions_in_range(self) -> None:
        resp = client.post(
            "/api/v1/resumes/job-specific-ats",
            json={"resume": _RESUME, "job_description": _JOB},
        )
        coverage = resp.json()["coverage"]
        for key in ("required_coverage", "preferred_coverage", "overall_coverage"):
            value = coverage[key]
            if value is not None:
                assert 0.0 <= value <= 1.0


class TestJobSpecificEndpointErrors:
    def test_empty_body_returns_422(self) -> None:
        resp = client.post("/api/v1/resumes/job-specific-ats", json={})
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"

    def test_missing_job_description_returns_422(self) -> None:
        resp = client.post(
            "/api/v1/resumes/job-specific-ats", json={"resume": _RESUME}
        )
        assert resp.status_code == 422

    def test_error_never_echoes_input(self) -> None:
        resp = client.post(
            "/api/v1/resumes/job-specific-ats",
            json={"resume": "secret_marker_6b", "job_description": _JOB},
        )
        body = resp.json()
        assert "secret_marker_6b" not in body.get("error", {}).get("message", "")

    def test_health_endpoint_unchanged(self) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
