"""Orchestrator for the Resume Copilot (Phase 7A).

Responsibilities:
- validate every request against the operation contract and resource bounds,
- resolve ``target_ref`` into ``target_text`` when the latter is absent,
- run providers in a fixed priority order with transparent fallback,
- never log or return prompts, responses, resume/job text, or PII.

Provider order is ``[LocalOllamaProvider, DeterministicFallbackProvider]``.
Because the deterministic provider is always available, an "all providers
failed" state is unreachable in practice; it is still raised as
``ProviderUnavailableError`` so callers never see a silent failure.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.copilot.config import COPILOT_IMPLEMENTATION_VERSION, copilot_settings
from app.copilot.editing import (
    attach_edit_proposals,
    build_target_clarification,
    edit_target_options,
    infer_edit_target,
    reconcile_description_edit,
)
from app.copilot.errors import (
    CopilotError,
    InvalidRequestError,
    ProviderUnavailableError,
    UnsupportedOperationError,
)
from app.copilot.evidence import resolve_target
from app.copilot.operations import get_contract, requires_target
from app.copilot.providers import DeterministicFallbackProvider, LocalOllamaProvider
from app.copilot.providers.base import CopilotProvider
from app.copilot.schemas import (
    CopilotEditTarget,
    CopilotOperation,
    CopilotRequest,
    CopilotResponse,
    CopilotStatus,
    ProviderKind,
    ProviderStatus,
)

_KNOWN_OPERATIONS = frozenset(op for op in CopilotOperation)


class CopilotService:
    """Front-door for Copilot operations; used by the API layer."""

    def __init__(
        self,
        providers: Sequence[CopilotProvider] | None = None,
    ) -> None:
        """Create the service, optionally injecting the provider chain.

        The default chain is ``[LocalOllamaProvider, DeterministicFallbackProvider]``.
        Tests inject a deterministic-only or fake chain so no local LLM is
        invoked.
        """
        self._providers: tuple[CopilotProvider, ...] = (
            tuple(providers)
            if providers is not None
            else (LocalOllamaProvider(), DeterministicFallbackProvider())
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(self, request: CopilotRequest) -> CopilotResponse:
        """Validate a request, run the best available provider, return the response.

        Replacement-text suggestions are post-processed into structured edit
        proposals (Phase 7C) after the provider returns, so both the LLM and the
        deterministic fallback always go through the same allowlist and
        fact-preservation checks. For free-form requests without an explicit
        ``target_ref``, a safe target is inferred up front so the provider is
        told exactly which text to rewrite.
        """
        resolved = self._with_inferred_target(request)
        resolved = self._with_resolved_target(resolved)
        self._validate(resolved)
        options = edit_target_options(resolved)
        if options:
            return self._clarification_response(resolved, options)
        response = self._run(resolved)
        response = attach_edit_proposals(resolved, response)
        return reconcile_description_edit(resolved, response)

    def status(self) -> CopilotStatus:
        """Return the current availability of all Copilot providers."""
        providers = [
            ProviderStatus(
                provider=meta.provider,
                provider_label=meta.provider_label,
                model=meta.model,
                available=meta.available,
                note=meta.note,
            )
            for meta in (p.metadata() for p in self._providers)
        ]
        fallback_available = any(
            p.provider_kind() == ProviderKind.DETERMINISTIC and p.available()
            for p in self._providers
        )
        return CopilotStatus(
            providers=providers,
            fallback_available=fallback_available,
            version=COPILOT_IMPLEMENTATION_VERSION,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _clarification_response(
        self, request: CopilotRequest, options: list[CopilotEditTarget]
    ) -> CopilotResponse:
        """Answer an ambiguous edit request with an explicit target choice.

        The deterministic provider passively supplies the advisory body so the
        response still carries provider metadata and a disclaimer, but the
        clarification carries the concrete options and no LLM generation is
        spent on a request the system cannot safely resolve.
        """
        clarification = build_target_clarification(request, options)
        advisory = DeterministicFallbackProvider().run(request)
        return advisory.model_copy(update={"clarification": clarification})

    def _run(self, request: CopilotRequest) -> CopilotResponse:
        last_error = ""
        for provider in self._ordered_providers():
            if not provider.supports(request):
                continue
            if not provider.available():
                continue
            try:
                return provider.run(request)
            except CopilotError as exc:
                last_error = str(exc)
                continue
        raise ProviderUnavailableError(
            "No Copilot provider could complete the request."
            + (f" Last provider error: {last_error}" if last_error else "")
        )

    def _ordered_providers(self) -> list[CopilotProvider]:
        return list(self._providers)

    def _with_inferred_target(self, request: CopilotRequest) -> CopilotRequest:
        """Resolve an editable target for free-form requests from their wording.

        An explicit client ``target_ref`` always wins (the "Act on (optional)"
        override). Otherwise, when the request clearly names one editable
        resume value, the inferred ``target_ref`` is injected so providers
        receive the exact target text and edit proposals attach to it.
        """
        if request.operation != CopilotOperation.FREE_FORM:
            return request
        if (request.target_ref or "").strip():
            return request
        inferred = infer_edit_target(
            request.resume, (request.user_request or "").strip()
        )
        if inferred is None:
            return request
        return request.model_copy(update={"target_ref": inferred.path})

    def _with_resolved_target(self, request: CopilotRequest) -> CopilotRequest:
        target_text = request.target_text or ""
        if not target_text and request.target_ref:
            resolved = resolve_target(request.resume, request.target_ref)
            target_text = resolved or ""
        return request.model_copy(update={"target_text": target_text.strip()})

    def _validate(self, request: CopilotRequest) -> None:
        if request.operation not in _KNOWN_OPERATIONS:
            raise UnsupportedOperationError(
                f"Unsupported Copilot operation: {request.operation!r}."
            )
        contract = get_contract(request.operation)

        if contract.requires_job and request.job_description is None:
            raise InvalidRequestError(
                f"Operation '{request.operation.value}' requires a job_description."
            )
        if (
            request.operation == CopilotOperation.EXPLAIN_FINDING
            and not (request.finding_ref or "").strip()
        ):
            raise InvalidRequestError(
                "Operation 'explain_finding' requires a 'finding_ref'."
            )
        if (
            requires_target(request.operation)
            and not (request.target_text or "").strip()
        ):
            raise InvalidRequestError(
                "Operation 'improve_bullet' requires a non-empty 'target_ref' "
                "or 'target_text'."
            )
        user_request = (request.user_request or "").strip()
        if request.operation == CopilotOperation.FREE_FORM:
            if not user_request:
                raise InvalidRequestError(
                    "Operation 'free-form' requires a non-empty 'user_request'."
                )
            if len(user_request) > copilot_settings.max_request_chars:
                raise InvalidRequestError(
                    "The free-text request exceeds the configured limit of "
                    f"{copilot_settings.max_request_chars} characters."
                )
        if len(request.target_text or "") > copilot_settings.max_target_text_chars:
            raise InvalidRequestError(
                "The target text exceeds the configured limit of "
                f"{copilot_settings.max_target_text_chars} characters."
            )
