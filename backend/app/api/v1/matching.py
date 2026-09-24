"""Matching endpoints: deterministic Baseline Match Score for Resume vs Job."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.hybrid_matching.service import compute_hybrid_match
from app.job_parsing.schemas import JobDescription
from app.matching.service import match_resume_to_job
from app.parsing.schemas import Resume
from app.semantic_matching.model import ModelLoadError, SemanticError
from app.semantic_matching.service import compute_semantic_match

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/matching", tags=["matching"])


class MatchRequestBody(BaseModel):
    """Transient request: a fully parsed Resume and JobDescription."""

    resume: Resume
    job: JobDescription


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


@router.post("/score")
async def score_matching(body: MatchRequestBody) -> JSONResponse:
    """Score a resume against a job description (deterministic, transient).

    No persistence, no files, no external calls: the request and result exist
    only for the lifetime of this call.
    """
    try:
        result = match_resume_to_job(resume=body.resume, job=body.job)
    except Exception:
        logger.exception("Unexpected error during matching")
        return _error_response(
            500,
            "matching_failed",
            "An unexpected error occurred.",
        )
    return JSONResponse(status_code=200, content=result.model_dump(mode="json"))


@router.post("/hybrid")
async def hybrid_matching(body: MatchRequestBody) -> JSONResponse:
    """Hybrid Match Score: authoritative deterministic signal composed with
    the local semantic layer.

    Transient only. The endpoint degrades gracefully: if the local semantic
    model is unavailable or fails, the deterministic signal alone is returned
    with the semantic availability marked. No persistence, no files, no
    external calls.
    """
    try:
        result = compute_hybrid_match(resume=body.resume, job=body.job)
    except Exception:
        logger.exception("Unexpected error during hybrid matching")
        return _error_response(
            500,
            "hybrid_matching_failed",
            "An unexpected error occurred.",
        )
    return JSONResponse(status_code=200, content=result.model_dump(mode="json"))


@router.post("/semantic")
async def semantic_matching(body: MatchRequestBody) -> JSONResponse:
    """Local semantic similarity between a resume and a job description.

    Local inference only: no external service, no persistence, no files. The
    underlying model is loaded lazily on first use and cached in-process.
    """
    try:
        result = compute_semantic_match(resume=body.resume, job=body.job)
    except ModelLoadError:
        logger.warning("Semantic matching unavailable")
        return _error_response(
            503,
            "semantic_model_unavailable",
            "The local semantic model is unavailable. "
            "Reinstall dependencies with `pip install onnxruntime tokenizers` "
            "to enable semantic matching.",
        )
    except SemanticError:
        logger.warning("Semantic matching failed")
        return _error_response(
            500,
            "semantic_inference_failed",
            "Local semantic inference failed.",
        )
    except Exception:
        logger.exception("Unexpected error during semantic matching")
        return _error_response(
            500,
            "semantic_matching_failed",
            "An unexpected error occurred.",
        )
    return JSONResponse(status_code=200, content=result.model_dump(mode="json"))
