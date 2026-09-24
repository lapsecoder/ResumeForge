"""Provider protocol for the Resume Copilot (Phase 7A).

The Copilot is a suggestion engine, not an autonomous agent. Every provider:
- answers a bounded availability check,
- executes exactly one controlled operation and returns a structured response,
- exposes safe, non-sensitive provider metadata,
- fails gracefully (raising ``CopilotError`` subclasses) rather than returning
  malformed or invented results.

Providers never log prompts, responses, resume text, job text, PII, or
generated suggestions.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.copilot.schemas import (
    CopilotRequest,
    CopilotResponse,
    ProviderKind,
    ProviderMetadata,
)


@runtime_checkable
class CopilotProvider(Protocol):
    """Anything able to answer a controlled Copilot request."""

    def provider_kind(self) -> ProviderKind: ...

    def available(self) -> bool:
        """Return True when this provider is usable right now.

        Availability checks must be cheap and must never raise; failures are
        reported as ``False`` so the orchestrator can move on.
        """
        ...

    def supports(self, request: CopilotRequest) -> bool:
        """Return True when this provider can lawfully run the request."""
        ...

    def metadata(self) -> ProviderMetadata:
        """Return safe metadata describing this provider's current state."""
        ...

    def run(self, request: CopilotRequest) -> CopilotResponse:
        """Execute the request and return a fully validated response.

        Raises ``CopilotError`` subclasses on any failure so the orchestrator
        can transparently degrade to the next provider.
        """
        ...
