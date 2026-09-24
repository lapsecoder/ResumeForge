"""Copilot providers for the Resume Copilot (Phase 7A).

Provider protocol: ``app.copilot.providers.base.CopilotProvider``.
Implementations: ``LocalOllamaProvider``, ``DeterministicFallbackProvider``.
"""

from app.copilot.providers.deterministic import DeterministicFallbackProvider
from app.copilot.providers.ollama import LocalOllamaProvider

__all__ = ["DeterministicFallbackProvider", "LocalOllamaProvider"]
