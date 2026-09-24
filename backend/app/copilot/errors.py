"""Exception hierarchy for the Resume Copilot.

Every error here is safe to surface: messages never embed user content
(resume/JD text, PII, or generated suggestions). The API layer maps these to
the standard ``{error: {...}}`` contract without stack traces.
"""

from __future__ import annotations


class CopilotError(Exception):
    """Base class for copilot failures (never includes user content)."""


class InvalidRequestError(CopilotError):
    """The request violates an operation contract or a resource bound."""


class ProviderUnavailableError(CopilotError):
    """No usable provider could answer the request (LLM absent + no fallback)."""


class ProviderTimeoutError(CopilotError):
    """A provider exceeded its allowed response time."""


class MalformedProviderOutputError(CopilotError):
    """A provider returned output that did not match the structured contract."""


class UnsupportedOperationError(CopilotError):
    """A provider was asked for an operation it does not support."""
