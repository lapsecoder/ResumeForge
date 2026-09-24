"""Configuration for the Resume Copilot.

Settings derive from environment variables with the ``COPILOT_`` prefix
(e.g. ``COPILOT_OLLAMA_MODEL``). The Copilot is a zero-cost, local-first,
suggestion engine: the default provider talks only to a local Ollama server,
and every input/output is size-bounded to protect the local machine and keep
responses deterministic.
"""

from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlparse

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Default local model. Explicit default for the local Ollama server on this
#: class of hardware (RTX 50-series, 8 GB VRAM); verified against the actual
#: installed models at runtime. The provider never downloads models.
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"

#: Copilot implementation version — bumped when the Copilot changes its public
#: contract or its fallback behaviour. 7B adds the factual-validation pipeline
#: in front of LLM output and fact-grounded deterministic drafts. 7C adds
#: structured, allowlisted edit proposals with an explicit apply/revert model.
#: 7D adds bounded free-text requests (operation ``free-form``) that flow
#: through the same generation → validation → evidence → edit pipeline as the
#: controlled operations.
COPILOT_IMPLEMENTATION_VERSION = "7d-copilot-1.0"

#: Default API disclaimer surfaced on every response.
COPILOT_DISCLAIMER = (
    "The ResumeForge Copilot provides resume-writing assistance: it explains "
    "issues, suggests wording, and helps you align existing evidence with a "
    "target role. It does not predict hiring outcomes, ATS pass rates, or "
    "recruiter decisions, and it never invents facts that are not already in "
    "the materials you provided."
)


class CopilotSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="COPILOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Master switch for the local LLM provider. When disabled the deterministic
    # fallback is the only provider, mirroring an "Ollama unavailable" state.
    llm_enabled: bool = True

    # Ollama server. Defaults to the local default port and is restricted to
    # loopback: a client can never point the Copilot at a remote endpoint.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = DEFAULT_OLLAMA_MODEL

    # Generation is performed with a bounded number of output tokens and a hard
    # timeout; clients cannot tune either.
    llm_request_timeout_seconds: float = 120.0
    llm_availability_timeout_seconds: float = 2.5
    # The structured JSON response (explanation + suggestions + evidence) needs
    # enough room to complete; a too-small cap truncates the JSON and forces a
    # fallback. 2048 comfortably fits a valid multi-suggestion response.
    llm_max_tokens: int = 2048

    # --- Resource bounds (input and output) ---
    # Target text sent for rewrite operations (e.g. one bullet).
    max_target_text_chars: int = 2000
    # Free-text request (operation ``free-form``) sent by the user.
    max_request_chars: int = 2000
    # Combined resume+job data rendered into an LLM prompt.
    max_prompt_chars: int = 12000
    # Total generated suggestion text accepted from a provider.
    max_response_chars: int = 12000
    # Maximum number of suggestions returned for one request.
    max_suggestions: int = 8
    # Max evidence references attached to one suggestion.
    max_evidence_per_suggestion: int = 5
    # Max length of a proposed edit value accepted or emitted (Phase 7C).
    max_edit_value_chars: int = 4000

    @field_validator("ollama_base_url")
    @classmethod
    def _loopback_only(cls, value: str) -> str:
        """Allow only localhost / loopback Ollama endpoints (http only)."""
        parsed = urlparse(value)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "http" or host not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError(
                "COPILOT_OLLAMA_BASE_URL must point at a local Ollama server "
                "(only http://localhost or http://127.0.0.1 are allowed)"
            )
        return value.rstrip("/")

    @field_validator(
        "llm_max_tokens",
        "max_target_text_chars",
        "max_request_chars",
        "max_prompt_chars",
        "max_response_chars",
        "max_suggestions",
        "max_evidence_per_suggestion",
        "max_edit_value_chars",
    )
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("copilot size bounds must be positive")
        return value


@lru_cache
def get_copilot_settings() -> CopilotSettings:
    """Return a cached CopilotSettings instance."""
    return CopilotSettings()


copilot_settings = get_copilot_settings()
