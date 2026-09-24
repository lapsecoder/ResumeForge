"""Role analysis endpoints.

Provides role-level compatibility analysis using local profiles
and the existing deterministic matching pipeline.

Transient and privacy-first: nothing is persisted, no content is
logged, and no external calls are made. Role profiles are local data.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.role_analysis.profiles import (
    SUPPORTED_ROLES,
    RoleProfile,
    search_roles,
)
from app.role_analysis.schemas import RoleAnalysisRequest
from app.role_analysis.service import analyze_role_compatibility

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/roles", tags=["roles"])


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


class RoleInfo(BaseModel):
    title: str
    aliases: list[str]


class SupportedRolesResponse(BaseModel):
    roles: list[RoleInfo]
    total: int


@router.get("/supported", response_model=SupportedRolesResponse)
async def supported_roles(
    q: str | None = Query(default=None, max_length=100),
) -> SupportedRolesResponse:
    """List supported role profiles for the searchable role selector.

    Pass ``q`` to filter by keyword (case-insensitive substring match
    against title and aliases). With no ``q``, all supported roles are
    returned. Local data only — no persistence, no external calls.
    """
    if q:
        matches: list[RoleProfile] = search_roles(q)
    else:
        matches = list(SUPPORTED_ROLES)
    return SupportedRolesResponse(
        roles=[RoleInfo(title=p.title, aliases=list(p.aliases)) for p in matches],
        total=len(matches),
    )


class RoleAnalysisBody(BaseModel):
    role_title: str
    resume: dict[str, object]


@router.post("/analyze")
async def analyze_role(body: RoleAnalysisBody) -> JSONResponse:
    """Role-level compatibility analysis (deterministic, transient).

    Resolves the role title to a local profile, runs the existing
    deterministic matching pipeline, and returns compatibility
    results, skills gaps, experience gaps, and recommendations.

    No persistence, no files, no external calls, no embeddings, no LLM.
    """
    try:
        from app.parsing.schemas import Resume

        resume = Resume.model_validate(body.resume)
        request = RoleAnalysisRequest(
            role_title=body.role_title,
            resume=resume,
        )
        result = analyze_role_compatibility(request)
    except Exception:
        logger.exception("Unexpected error during role analysis")
        return _error_response(
            500,
            "role_analysis_failed",
            "An unexpected error occurred.",
        )
    return JSONResponse(status_code=200, content=result.model_dump(mode="json"))
