"""Resume Copilot foundation (Phase 7A).

Controlled, privacy-first, deterministic-first suggestions layer.
"""

from app.copilot.config import (
    COPILOT_DISCLAIMER,
    COPILOT_IMPLEMENTATION_VERSION,
    CopilotSettings,
    copilot_settings,
)
from app.copilot.errors import (
    CopilotError,
    InvalidRequestError,
    MalformedProviderOutputError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    UnsupportedOperationError,
)
from app.copilot.schemas import (
    CopilotAnalysisContext,
    CopilotEvidence,
    CopilotOperation,
    CopilotRequest,
    CopilotResponse,
    CopilotStatus,
    EvidenceKind,
    ProviderKind,
    ProviderMetadata,
    SuggestionCategory,
    VerificationLevel,
)
from app.copilot.service import CopilotService

__all__ = [
    "COPILOT_DISCLAIMER",
    "COPILOT_IMPLEMENTATION_VERSION",
    "CopilotAnalysisContext",
    "CopilotError",
    "CopilotEvidence",
    "CopilotOperation",
    "CopilotRequest",
    "CopilotResponse",
    "CopilotService",
    "CopilotSettings",
    "CopilotStatus",
    "EvidenceKind",
    "InvalidRequestError",
    "MalformedProviderOutputError",
    "ProviderKind",
    "ProviderMetadata",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "SuggestionCategory",
    "UnsupportedOperationError",
    "VerificationLevel",
    "copilot_settings",
]
