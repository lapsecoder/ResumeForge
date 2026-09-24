"""API tests for the ATS readiness analysis endpoint.

End-to-end with a synthetic resume JSON body. All processing is transient:
no database, no files, no external services.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_FULL_RESUME = {
    "contact": {
        "name": "Jane Doe",
        "email": "jane@example.com",
        "phone": "123-456-7890",
        "linkedin": "https://linkedin.com/in/jane",
        "github": "https://github.com/jane",
    },
    "summary": "Experienced software engineer.",
    "experience": [
        {
            "company": "Acme",
            "title": "Senior Engineer",
            "start_date": "Jan 2020",
            "end_date": "Present",
            "description": "Platform team.",
            "achievements": [
                "Built a parsing service serving 10,000 users.",
                "Reduced processing time by 35%.",
                "Led a team of 6 engineers.",
            ],
        }
    ],
    "education": [
        {
            "institution": "IIT",
            "degree": "B.Tech",
            "field": "Computer Science",
            "start_date": "2014",
            "end_date": "2018",
        }
    ],
    "skills": {
        "technical": ["Python", "FastAPI", "PostgreSQL", "Docker"],
        "tools": ["Git", "CI/CD"],
        "languages": ["English"],
        "soft": ["Leadership"],
        "all": [],
    },
    "projects": [
        {
            "name": "ResumeForge",
            "description": "Resume parsing service used by 5,000 users.",
            "technologies": ["Python", "FastAPI"],
        }
    ],
    "certifications": [
        {
            "name": "AWS Certified Developer",
            "issuer": "Amazon",
            "date": "2021",
        }
    ],
    "custom_sections": [],
    "metadata": {
        "word_count": 600,
        "file_type": "pdf",
        "overall_confidence": "high",
        "section_confidence": [],
        "section_order": [
            "header",
            "summary",
            "experience",
            "education",
            "skills",
            "projects",
            "certifications",
        ],
    },
}


class TestATSEndpointSuccess:
    def test_returns_200_with_valid_resume(self) -> None:
        resp = client.post(
            "/api/v1/resumes/ats-analysis",
            json={"resume": _FULL_RESUME},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "overall_score" in body
        assert "score_label" in body
        assert "category_scores" in body
        assert "findings" in body
        assert "metadata" in body

    def test_score_is_bounded(self) -> None:
        resp = client.post(
            "/api/v1/resumes/ats-analysis",
            json={"resume": _FULL_RESUME},
        )
        body = resp.json()
        assert 0.0 <= body["overall_score"] <= 100.0
        for cs in body["category_scores"]:
            if cs["score"] is not None:
                assert 0.0 <= cs["score"] <= 100.0

    def test_metadata_contains_disclaimer(self) -> None:
        resp = client.post(
            "/api/v1/resumes/ats-analysis",
            json={"resume": _FULL_RESUME},
        )
        body = resp.json()
        disclaimer = body["metadata"]["disclaimer"].lower()
        assert "not a probability" in disclaimer
        assert "heuristic" in disclaimer

    def test_findings_have_required_fields(self) -> None:
        resp = client.post(
            "/api/v1/resumes/ats-analysis",
            json={"resume": _FULL_RESUME},
        )
        body = resp.json()
        for f in body["findings"]:
            assert "category" in f
            assert "severity" in f
            assert "rule_id" in f
            assert "title" in f
            assert "explanation" in f
            assert "recommendation" in f
            assert "impact" in f

    def test_sparse_resume_scores_lower(self) -> None:
        sparse = {**_FULL_RESUME, "experience": [], "summary": None}
        sparse["metadata"] = {
            **_FULL_RESUME["metadata"],
            "word_count": 40,
        }
        full_resp = client.post(
            "/api/v1/resumes/ats-analysis",
            json={"resume": _FULL_RESUME},
        )
        sparse_resp = client.post(
            "/api/v1/resumes/ats-analysis",
            json={"resume": sparse},
        )
        assert sparse_resp.json()["overall_score"] <= full_resp.json()["overall_score"]

    def test_deterministic_repeat(self) -> None:
        r1 = client.post(
            "/api/v1/resumes/ats-analysis",
            json={"resume": _FULL_RESUME},
        )
        r2 = client.post(
            "/api/v1/resumes/ats-analysis",
            json={"resume": _FULL_RESUME},
        )
        assert r1.json() == r2.json()


class TestATSEndpointErrors:
    def test_empty_body_returns_422(self) -> None:
        resp = client.post("/api/v1/resumes/ats-analysis", json={})
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["code"] == "validation_error"

    def test_invalid_metadata_type_returns_422(self) -> None:
        bad_resume = {**_FULL_RESUME, "metadata": "not_a_dict"}
        resp = client.post(
            "/api/v1/resumes/ats-analysis", json={"resume": bad_resume}
        )
        assert resp.status_code == 422

    def test_error_payload_never_echoes_input(self) -> None:
        resp = client.post(
            "/api/v1/resumes/ats-analysis", json={"resume": {}}
        )
        body = resp.json()
        assert "secret_marker" not in body.get("error", {}).get("message", "")


class TestATSEndpointHealthUnchanged:
    def test_health_endpoint_still_works(self) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
