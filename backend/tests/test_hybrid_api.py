"""API tests for POST /api/v1/matching/hybrid (Phase 5C).

Verifies the transient, explainable HybridMatchResult; graceful degradation
when the semantic model is missing or fails; structured validation errors; and
that neither the deterministic nor the semantic signals leak raw embeddings or
echo input narrative. The provider singleton is monkeypatched so no model is
downloaded. ``/score`` and ``/semantic`` are asserted to be unaffected.
"""

from __future__ import annotations

from typing import Any

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


def _post_hybrid(body: dict) -> Any:
    return client.post("/api/v1/matching/hybrid", json=body)


class TestHybridSuccess:
    def test_returns_complete_hybrid_result(
        self, fake_provider: ApiFakeProvider
    ) -> None:
        resp = _post_hybrid({"resume": RESUME_BODY, "job": JOB_BODY})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "error" not in body

        assert body["overall_score"] == 100.0
        assert body["deterministic"]["metadata"]["method"] == "deterministic-baseline"
        assert body["semantic"] is not None
        assert body["semantic"]["metadata"]["method"] == "local-semantic-embedding"
        assert body["semantic"]["overall_similarity"] == 1.0

        components = body["component_scores"]
        assert components["deterministic_overall"] == 100.0
        assert components["semantic_overall"] == 1.0
        assert components["hybrid_overall"] == 100.0

        assert isinstance(body["semantic_insights"], list)
        assert any(i["category"] == "summary" for i in body["semantic_insights"])

        metadata = body["metadata"]
        assert metadata["method"] == "hybrid-match"
        assert metadata["label"] == "Hybrid Match Score"
        assert metadata["version"] == "5c-hybrid-1.0"
        assert metadata["mode"] == "hybrid"
        assert metadata["weights"] == {"deterministic": 0.70, "semantic": 0.30}
        assert metadata["evidence_quality"] == "high"
        assert metadata["semantic_availability"]["status"] == "available"
        assert metadata["semantic_metadata"]["model_name"] == "fake-api-model"
        assert "heuristic relevance score" in metadata["note"]

    def test_no_embedding_vectors_in_response(
        self, fake_provider: ApiFakeProvider
    ) -> None:
        resp = _post_hybrid({"resume": RESUME_BODY, "job": JOB_BODY})
        body = resp.json()
        stack = [body]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                assert "embedding" not in {str(k).lower() for k in node}
                assert "vector" not in {str(k).lower() for k in node}
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)

    def test_single_batch_per_side(self, fake_provider: ApiFakeProvider) -> None:
        _post_hybrid({"resume": RESUME_BODY, "job": JOB_BODY})
        assert fake_provider.embed_calls == 2

    def test_input_narrative_not_echoed_on_success(
        self, fake_provider: ApiFakeProvider
    ) -> None:
        secret = {
            "resume": {
                **RESUME_BODY,
                "contact": {"name": "SUPER-SECRET-NAME", "email": "a@b.com"},
                "summary": "Very SECRET-PHRASE-SUMMARY.",
            }
        }
        resp = _post_hybrid({"resume": secret["resume"], "job": JOB_BODY})
        assert resp.status_code == 200
        assert "SUPER-SECRET-NAME" not in resp.text
        assert "SECRET-PHRASE-SUMMARY" not in resp.text


class TestGracefulDegradation:
    def test_model_unavailable_returns_deterministic_200(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _broken(*args, **kwargs):
            raise ModelLoadError("missing library")

        monkeypatch.setattr(semantic_service, "get_embedding_provider", _broken)
        resp = _post_hybrid({"resume": RESUME_BODY, "job": JOB_BODY})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["overall_score"] == 100.0
        assert body["semantic"] is None
        assert body["metadata"]["mode"] == "deterministic-only"
        assert (
            body["metadata"]["semantic_availability"]["status"] == "model_unavailable"
        )
        assert body["metadata"]["semantic_metadata"] is None

    def test_inference_failure_returns_deterministic_200(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _broken(*args, **kwargs):
            raise InferenceError("mock failure")

        monkeypatch.setattr(semantic_service, "get_embedding_provider", _broken)
        resp = _post_hybrid({"resume": RESUME_BODY, "job": JOB_BODY})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["metadata"]["mode"] == "deterministic-only"
        assert body["metadata"]["semantic_availability"]["status"] == "inference_failed"

    def test_unexpected_semantic_failure_degrades_gracefully(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _broken(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(semantic_service, "get_embedding_provider", _broken)
        resp = _post_hybrid({"resume": RESUME_BODY, "job": JOB_BODY})
        assert resp.status_code == 200, resp.text
        assert resp.json()["metadata"]["mode"] == "deterministic-only"
        assert resp.json()["metadata"]["semantic_availability"]["status"] == "error"

    def test_deterministic_signal_survives_semantic_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _broken(*args, **kwargs):
            raise ModelLoadError("missing library")

        monkeypatch.setattr(semantic_service, "get_embedding_provider", _broken)
        body = _post_hybrid({"resume": RESUME_BODY, "job": JOB_BODY}).json()
        assert body["deterministic"]["overall_score"] == 100.0
        assert "Required skill matched: Python" in body["matched_requirements"]


class TestValidationErrors:
    def _assert_structured_validation_error(self, resp) -> None:
        assert resp.status_code == 422
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == "validation_error"
        assert body["error"]["message"]
        assert body["error"]["request_id"]

    def test_missing_job_field(self, fake_provider: ApiFakeProvider) -> None:
        self._assert_structured_validation_error(
            _post_hybrid({"resume": RESUME_BODY})
        )

    def test_missing_resume_field(self, fake_provider: ApiFakeProvider) -> None:
        self._assert_structured_validation_error(_post_hybrid({"job": JOB_BODY}))

    def test_empty_body(self, fake_provider: ApiFakeProvider) -> None:
        self._assert_structured_validation_error(_post_hybrid({}))

    def test_input_never_echoed_on_validation(
        self, fake_provider: ApiFakeProvider
    ) -> None:
        secret = {
            "resume": {"skills": {"all": ["TOP-SECRET-SKILL"]}},
            "job": JOB_BODY,
        }
        resp = _post_hybrid(secret)
        assert resp.status_code == 422
        assert "TOP-SECRET-SKILL" not in resp.text


class TestOtherEndpointsUnaffected:
    def test_score_endpoint_still_returns_baseline(self) -> None:
        resp = client.post(
            "/api/v1/matching/score", json={"resume": RESUME_BODY, "job": JOB_BODY}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["metadata"]["method"] == "deterministic-baseline"

    def test_semantic_endpoint_still_returns_similarity(
        self, fake_provider: ApiFakeProvider
    ) -> None:
        resp = client.post(
            "/api/v1/matching/semantic", json={"resume": RESUME_BODY, "job": JOB_BODY}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["metadata"]["method"] == "local-semantic-embedding"
