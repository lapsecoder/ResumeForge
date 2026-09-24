"""Copilot endpoints: controlled resume-writing suggestions (Phase 7A).

All payloads are transient and stateless: nothing is persisted, no prompts or
responses are logged, and no PII is ever emitted. The LLM (local Ollama) is
best-effort; the deterministic fallback guarantees a usable response whenever
the request contract is satisfiable.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.copilot.errors import (
    InvalidRequestError,
    ProviderUnavailableError,
    UnsupportedOperationError,
)
from app.copilot.schemas import CopilotRequest
from app.copilot.service import CopilotService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/copilot", tags=["copilot"])

_service = CopilotService()


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": str(uuid.uuid4()),
            }
        },
    )


@router.post("/suggest")
def copilot_suggest(payload: CopilotRequest) -> JSONResponse:
    """Run one controlled Copilot operation and return structured suggestions."""
    operation = payload.operation.value
    try:
        result = _service.suggest(payload)
    except InvalidRequestError as exc:
        logger.info("copilot invalid_request operation=%s error=%s", operation, exc)
        return _error_response(422, "invalid_request", str(exc))
    except UnsupportedOperationError as exc:
        logger.info(
            "copilot unsupported_operation operation=%s error=%s", operation, exc
        )
        return _error_response(400, "unsupported_operation", str(exc))
    except ProviderUnavailableError as exc:
        logger.warning("copilot unavailable operation=%s error=%s", operation, exc)
        return _error_response(
            503,
            "copilot_unavailable",
            "Copilot suggestions are currently unavailable.",
        )
    except Exception as exc:
        logger.warning("copilot unexpected operation=%s error=%r", operation, exc)
        return _error_response(
            500,
            "copilot_failed",
            "An unexpected error occurred while generating suggestions.",
        )
    return JSONResponse(status_code=200, content=result.model_dump(mode="json"))


@router.get("/status")
def copilot_status() -> JSONResponse:
    """Report which Copilot providers are currently available."""
    content = _service.status().model_dump(mode="json")
    return JSONResponse(status_code=200, content=content)
