"""Provider interface and fallback-chain tests for the Resume Copilot.

Offline: the Ollama provider is exercised through a monkeypatched HTTP layer
only; no real network call, no local model, no GPU.
"""

from __future__ import annotations

from typing import Any

import pytest
from copilot_factories import make_job, make_resume

from app.copilot.config import CopilotSettings
from app.copilot.errors import (
    MalformedProviderOutputError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.copilot.providers import (
    DeterministicFallbackProvider,
    LocalOllamaProvider,
)
from app.copilot.providers.base import CopilotProvider
from app.copilot.schemas import (
    CopilotOperation,
    CopilotRequest,
    CopilotResponse,
    ProviderKind,
)
from app.copilot.service import CopilotService

VALID_LLM_OUTPUT = {
    "message": {
        "role": "assistant",
        "content": (
            '{"explanation": "Tightened wording.", "suggestions": ['
            '{"category": "clarity", "original_text": "I built stuff", '
            '"suggested_text": "Built stuff", "rationale": "Cleaner opening.", '
            '"verification": "verified", "requires_user_confirmation": false, '
            '"evidence": [{"kind": "fact", "source": "copilot.inference", '
            '"statement": "Preserved.", "reference": ""}]}]}'
        ),
    }
}


def _settings() -> CopilotSettings:
    return CopilotSettings(
        llm_enabled=True,
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen2.5-coder:7b",
    )


def _capture_provider(
    body_out: object, request_body: list[dict[str, Any]]
) -> LocalOllamaProvider:
    settings = _settings()
    provider = LocalOllamaProvider(settings)

    def fake_request(
        path: str,
        *,
        method: str,
        body: dict[str, Any] | None = None,
        timeout_seconds: float,
    ) -> Any:
        request_body.append(
            {**(body or {}), "_path": path, "_timeout": timeout_seconds}
        )
        if path.endswith("/api/tags"):
            known_models = (
                body_out.get("models") if isinstance(body_out, dict) else None
            )
            return {
                "models": known_models
                if known_models is not None
                else [{"name": settings.ollama_model}]
            }
        return body_out

    provider._request = fake_request  # type: ignore[method-assign]
    return provider


class _Raises(CopilotProvider):
    """Fake LLM-side provider that always fails with a CopilotError subclass."""

    def __init__(self, error: type[Exception] | None = None) -> None:
        self._error = error or ProviderUnavailableError("boom")

    def provider_kind(self) -> ProviderKind:
        return ProviderKind.LOCAL_OLLAMA

    def available(self) -> bool:
        return True

    def supports(self, request: CopilotRequest) -> bool:
        return True

    def metadata(self):  # type: ignore[no-untyped-def]
        from app.copilot.schemas import ProviderMetadata

        return ProviderMetadata(
            provider=ProviderKind.LOCAL_OLLAMA,
            provider_label="fake",
            available=True,
            fallback_used=False,
            version="test",
        )

    def run(self, request: CopilotRequest) -> CopilotResponse:
        raise self._error


class _DeterministicProxy(CopilotProvider):
    """Deterministic provider reachable through the protocol."""

    def __init__(self) -> None:
        self._inner = DeterministicFallbackProvider()

    def provider_kind(self) -> ProviderKind:
        return self._inner.provider_kind()

    def available(self) -> bool:
        return self._inner.available()

    def supports(self, request: CopilotRequest) -> bool:
        return self._inner.supports(request)

    def metadata(self):  # type: ignore[no-untyped-def]
        return self._inner.metadata()

    def run(self, request: CopilotRequest) -> CopilotResponse:
        return self._inner.run(request)


def _bullet_request() -> CopilotRequest:
    return CopilotRequest(
        operation=CopilotOperation.IMPROVE_BULLET,
        resume=make_resume(),
        target_text="I built stuff.",
    )


# ---------------------------------------------------------------------------
# Provider protocol conformance
# ---------------------------------------------------------------------------


def test_providers_satisfy_the_copilot_provider_protocol() -> None:
    assert isinstance(DeterministicFallbackProvider(), CopilotProvider)
    assert isinstance(LocalOllamaProvider(_settings()), CopilotProvider)
    assert isinstance(_Raises(), CopilotProvider)


# ---------------------------------------------------------------------------
# Availability checks (never raise)
# ---------------------------------------------------------------------------


def test_ollama_availability_is_false_when_disabled() -> None:
    provider = LocalOllamaProvider(
        CopilotSettings(llm_enabled=False, ollama_base_url="http://localhost:11434")
    )
    assert provider.available() is False


def test_ollama_availability_is_false_when_model_missing() -> None:
    provider = _capture_provider({"models": [{"name": "other-model"}]}, [])
    assert provider.available() is False


def test_ollama_availability_is_true_when_model_present() -> None:
    provider = _capture_provider({"models": [{"name": "qwen2.5-coder:7b"}]}, [])
    assert provider.available() is True


def test_ollama_availability_tolerates_network_failure() -> None:
    provider = LocalOllamaProvider(_settings())

    def boom(
        _path: str,
        *,
        method: str,
        body: dict[str, Any] | None = None,
        timeout_seconds: float,
    ) -> Any:
        raise OSError("connection refused")

    provider._request = boom  # type: ignore[method-assign]
    assert provider.available() is False


def test_ollama_supports_only_known_generative_operations() -> None:
    provider = LocalOllamaProvider(_settings())
    for operation in CopilotOperation:
        request = CopilotRequest(operation=operation, resume=make_resume())
        if operation == CopilotOperation.JOB_ALIGNMENT:
            request = request.model_copy(update={"job_description": make_job()})
        assert provider.supports(request) is True


# ---------------------------------------------------------------------------
# run() error mapping
# ---------------------------------------------------------------------------


def test_ollama_run_maps_timeout_to_provider_timeout() -> None:
    provider = LocalOllamaProvider(_settings())

    def slow(
        _path: str,
        *,
        method: str,
        body: dict[str, Any] | None = None,
        timeout_seconds: float,
    ) -> Any:
        raise TimeoutError()

    provider._request = slow  # type: ignore[method-assign]
    with pytest.raises(ProviderTimeoutError):
        provider.run(_bullet_request())


def test_ollama_run_maps_unreachable_to_unavailable() -> None:
    provider = LocalOllamaProvider(_settings())

    def down(
        _path: str,
        *,
        method: str,
        body: dict[str, Any] | None = None,
        timeout_seconds: float,
    ) -> Any:
        raise ProviderUnavailableError("mock network down")

    provider._request = down  # type: ignore[method-assign]
    with pytest.raises(ProviderUnavailableError):
        provider.run(_bullet_request())


def test_ollama_run_rejects_malformed_shapes() -> None:
    provider = _capture_provider(
        {"message": {"role": "assistant", "content": "not json"}}, []
    )
    with pytest.raises(MalformedProviderOutputError):
        provider.run(_bullet_request())


def test_ollama_run_rejects_valid_json_without_suggestions() -> None:
    provider = _capture_provider(
        {
            "message": {
                "role": "assistant",
                "content": '{"explanation": "noop", "suggestions": []}',
            }
        },
        [],
    )
    with pytest.raises(MalformedProviderOutputError):
        provider.run(_bullet_request())


def test_ollama_run_returns_structured_response() -> None:
    captured: list[dict[str, Any]] = []
    provider = _capture_provider(VALID_LLM_OUTPUT, captured)
    resp = provider.run(_bullet_request())
    assert resp.operation == CopilotOperation.IMPROVE_BULLET
    assert resp.suggestions[0].suggested_text == "Built stuff"
    assert resp.provider.provider == ProviderKind.LOCAL_OLLAMA
    assert resp.provider.model == "qwen2.5-coder:7b"
    assert resp.provider.fallback_used is False


def test_ollama_run_coerces_off_spec_category_without_discarding_response() -> None:
    captured: list[dict[str, Any]] = []
    provider = _capture_provider(
        {
            "message": {
                "role": "assistant",
                "content": (
                    '{"explanation": "Rewrite.", "suggestions": ['
                    '{"category": "improvement", "original_text": "Built stuff", '
                    '"suggested_text": "Built stuff end to end.", "rationale": "r", '
                    '"verification": "maybe", "requires_user_confirmation": false, '
                    '"evidence": [{"kind": "hunch", "source": "s", '
                    '"statement": "t"}]}]}'
                ),
            }
        },
        captured,
    )
    resp = provider.run(_bullet_request())
    suggestion = resp.suggestions[0]
    assert suggestion.category.value == "clarity"
    assert suggestion.verification.value == "unverified"
    assert suggestion.evidence[0].kind.value == "suggestion"
    assert suggestion.suggested_text == "Built stuff end to end."


def test_ollama_free_form_sends_bounded_user_request_block() -> None:
    captured: list[dict[str, Any]] = []
    provider = _capture_provider(VALID_LLM_OUTPUT, captured)
    request = CopilotRequest(
        operation=CopilotOperation.FREE_FORM,
        resume=make_resume(),
        user_request="Make my summary more concise.",
    )
    resp = provider.run(request)
    assert resp.operation == CopilotOperation.FREE_FORM
    user_content = str(captured[0]["messages"][1]["content"])
    assert "<USER_REQUEST>" in user_content
    assert "</USER_REQUEST>" in user_content
    assert "Make my summary more concise." in user_content


def test_ollama_supports_free_form_when_enabled() -> None:
    assert _settings().llm_enabled is True
    provider = LocalOllamaProvider(_settings())
    request = CopilotRequest(
        operation=CopilotOperation.FREE_FORM,
        resume=make_resume(),
        user_request="Improve my summary.",
    )
    assert provider.supports(request) is True


def test_ollama_run_caps_suggestions() -> None:
    suggestion = (
        '{"category": "clarity", "original_text": "", "suggested_text": "", '
        '"rationale": "r", "verification": "advisory", '
        '"requires_user_confirmation": false, "evidence": []}'
    )
    content = (
        '{"explanation": "many", "suggestions": ['
        + ",".join(suggestion for _ in range(50))
        + "]}"
    )
    provider = _capture_provider(
        {"message": {"role": "assistant", "content": content}}, []
    )
    resp = provider.run(_bullet_request())
    assert len(resp.suggestions) <= CopilotSettings().max_suggestions


# ---------------------------------------------------------------------------
# Service fallback chain
# ---------------------------------------------------------------------------


def test_service_uses_llm_when_healthy() -> None:
    captured: list[dict[str, Any]] = []
    service = CopilotService(providers=[_capture_provider(VALID_LLM_OUTPUT, captured)])
    resp = service.suggest(_bullet_request())
    assert resp.provider.provider == ProviderKind.LOCAL_OLLAMA
    assert resp.suggestions[0].suggested_text == "Built stuff"


def test_service_falls_back_when_llm_unavailable() -> None:
    service = CopilotService(
        providers=[_Raises(ProviderUnavailableError("down")), _DeterministicProxy()]
    )
    resp = service.suggest(_bullet_request())
    assert resp.provider.provider == ProviderKind.DETERMINISTIC
    assert resp.provider.fallback_used is True


def test_service_falls_back_when_llm_output_malformed() -> None:
    service = CopilotService(
        providers=[
            _Raises(MalformedProviderOutputError("bad json")),
            _DeterministicProxy(),
        ]
    )
    resp = service.suggest(_bullet_request())
    assert resp.provider.provider == ProviderKind.DETERMINISTIC


def test_service_free_form_falls_back_when_llm_unavailable() -> None:
    service = CopilotService(providers=[_Raises(), _DeterministicProxy()])
    resp = service.suggest(
        CopilotRequest(
            operation=CopilotOperation.FREE_FORM,
            resume=make_resume(),
            user_request="Improve my summary.",
        )
    )
    assert resp.provider.provider == ProviderKind.DETERMINISTIC
    assert resp.provider.fallback_used is True


def test_service_raises_when_every_provider_fails() -> None:
    service = CopilotService(providers=[_Raises(ProviderUnavailableError("down"))])
    with pytest.raises(ProviderUnavailableError):
        service.suggest(_bullet_request())


def test_service_default_chain_prefers_ollama() -> None:
    service = CopilotService()
    kinds = [p.provider_kind() for p in service._providers]
    assert kinds == [ProviderKind.LOCAL_OLLAMA, ProviderKind.DETERMINISTIC]


def test_service_status_reports_fallback() -> None:
    service = CopilotService(providers=[_Raises(), _DeterministicProxy()])
    status = service.status()
    assert status.fallback_available is True
    assert {p.provider for p in status.providers} == {
        ProviderKind.LOCAL_OLLAMA,
        ProviderKind.DETERMINISTIC,
    }
    assert status.version
