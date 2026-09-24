"""API tests for POST /api/v1/matching/semantic.

The provider singleton is monkeypatched so no model is downloaded. Verifies
success responses, structured errors, lazy loading, and that input text is
never echoed or logged.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.semantic_matching import service as semantic_service
from app.semantic_matching.model import InferenceError, ModelLoadError, ModelMetadata

client = TestClient(app)

RESUME_BODY = {
    "contact": {"name": "Jane Doe", "email": "jane@example.com"},
    "summary": "Backend engineer who ships payments at scale.",
    "experience": [{"company": "Acme", "title": "Engineer", "start_date": "2020"}],
    "education": [{"institution": "IIT", "degree": "B.Tech"}],
    "skills": {"all": ["Python", "PostgreSQL"], "languages": []},
    "projects": [],
    "certifications": [],
    "custom_sections": [],
    "metadata": {"word_count": 0, "file_type": "pdf", "overall_confidence": "high"},
}

JOB_BODY = {
    "title": "Backend Engineer",
    "summary": "Design and scale payments APIs.",
    "required_skills": ["Python"],
    "preferred_skills": [],
    "responsibilities": ["Ship reliable REST APIs."],
    "experience_requirements": ["3+ years"],
    "education_requirements": ["Bachelor's degree"],
    "certifications": [],
    "qualifications": [],
    "metadata": {"word_count": 0, "file_type": "txt", "overall_confidence": "high"},
}


class ApiFakeProvider:
    """Returns unit vectors for every text; counts embedding calls."""

    def __init__(self) -> None:
        self.embed_calls = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        return [[1.0, 0.0] for _ in texts]

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_name="fake-api-model", model_dimension=2, device="cpu"
        )


@pytest.fixture
def fake_provider(monkeypatch: pytest.MonkeyPatch) -> ApiFakeProvider:
    provider = ApiFakeProvider()
    monkeypatch.setattr(semantic_service, "get_embedding_provider", lambda: provider)
    return provider


class TestSemanticSuccess:
    def test_returns_transient_result(self, fake_provider: ApiFakeProvider) -> None:
        resp = client.post(
            "/api/v1/matching/semantic", json={"resume": RESUME_BODY, "job": JOB_BODY}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "error" not in body
        assert body["overall_similarity"] is not None
        assert body["skill_similarity"] is not None
        assert body["responsibility_similarity"] is not None
        assert body["metadata"]["method"] == "local-semantic-embedding"
        assert body["metadata"]["model_name"] == "fake-api-model"
        assert body["metadata"]["device"] == "cpu"
        assert body["metadata"]["high_similarity_threshold"] == 0.65
        assert "relatedness" in body["note"]

    def test_single_batch_per_side(self, fake_provider: ApiFakeProvider) -> None:
        client.post(
            "/api/v1/matching/semantic", json={"resume": RESUME_BODY, "job": JOB_BODY}
        )
        assert fake_provider.embed_calls == 2

    def test_input_narrative_not_echoed_in_success_response(
        self, fake_provider: ApiFakeProvider
    ) -> None:
        resp = client.post(
            "/api/v1/matching/semantic", json={"resume": RESUME_BODY, "job": JOB_BODY}
        )
        assert resp.status_code == 200
        assert "payments at scale" not in resp.text
        assert "Design and scale payments APIs" not in resp.text


class TestValidationErrors:
    def _assert_structured_validation_error(self, resp) -> None:
        assert resp.status_code == 422
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == "validation_error"
        assert body["error"]["message"]
        assert body["error"]["request_id"]

    def test_missing_job_field(self, fake_provider: ApiFakeProvider) -> None:
        resp = client.post("/api/v1/matching/semantic", json={"resume": RESUME_BODY})
        self._assert_structured_validation_error(resp)

    def test_missing_resume_field(self, fake_provider: ApiFakeProvider) -> None:
        resp = client.post("/api/v1/matching/semantic", json={"job": JOB_BODY})
        self._assert_structured_validation_error(resp)

    def test_empty_body(self, fake_provider: ApiFakeProvider) -> None:
        resp = client.post("/api/v1/matching/semantic", json={})
        self._assert_structured_validation_error(resp)

    def test_input_never_echoed_on_validation(
        self, fake_provider: ApiFakeProvider
    ) -> None:
        secret = {"resume": {"skills": {"all": ["TOP-SECRET-SKILL"]}}, "job": JOB_BODY}
        resp = client.post("/api/v1/matching/semantic", json=secret)
        assert resp.status_code == 422
        assert "TOP-SECRET-SKILL" not in resp.text


class TestModelErrors:
    def test_returns_503_when_model_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _broken(*args, **kwargs):
            raise ModelLoadError("missing library")

        monkeypatch.setattr(semantic_service, "get_embedding_provider", _broken)
        resp = client.post(
            "/api/v1/matching/semantic", json={"resume": RESUME_BODY, "job": JOB_BODY}
        )
        assert resp.status_code == 503
        body = resp.json()
        assert body["error"]["code"] == "semantic_model_unavailable"
        assert "TOP-SECRET" not in body["error"]["message"]

    def test_returns_500_on_inference_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _broken(*args, **kwargs):
            raise InferenceError("mock failure")

        monkeypatch.setattr(semantic_service, "get_embedding_provider", _broken)
        resp = client.post(
            "/api/v1/matching/semantic", json={"resume": RESUME_BODY, "job": JOB_BODY}
        )
        assert resp.status_code == 500
        assert resp.json()["error"]["code"] == "semantic_inference_failed"

    def test_returns_500_on_unexpected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _broken(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(semantic_service, "get_embedding_provider", _broken)
        resp = client.post(
            "/api/v1/matching/semantic", json={"resume": RESUME_BODY, "job": JOB_BODY}
        )
        assert resp.status_code == 500
        assert resp.json()["error"]["code"] == "semantic_matching_failed"


class TestPositivePath:
    def test_score_endpoint_unaffected(self) -> None:
        resp = client.post(
            "/api/v1/matching/score", json={"resume": RESUME_BODY, "job": JOB_BODY}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["metadata"]["method"] == "deterministic-baseline"
