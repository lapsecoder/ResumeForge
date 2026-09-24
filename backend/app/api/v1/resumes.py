"""Resume endpoints: extraction, deterministic parsing, and ATS analysis.

Everything is transient: upload → validate → extract → normalise → (parse) →
return. No user data is stored on the server. The ATS analysis accepts an
already-parsed Resume as JSON and never touches the filesystem or database.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.ats_analysis.job_specific.service import analyze_job_specific_ats
from app.ats_analysis.service import analyze_resume
from app.ingestion.schemas import ExtractionResult
from app.ingestion.service import IngestionError, extract_resume_text
from app.ingestion.validators import ValidationError
from app.job_parsing.schemas import JobDescription
from app.parsing.parser import ParsingError, parse_resume
from app.parsing.schemas import Resume

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/resumes", tags=["resumes"])


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


@router.post("/extract")
async def extract_resume(
    file: Annotated[
        UploadFile,
        File(description="Resume file (PDF, DOCX, or TXT)"),
    ],
) -> JSONResponse:
    content = await file.read()
    size = len(content)

    try:
        result: ExtractionResult = extract_resume_text(
            filename=file.filename,
            content_type=file.content_type,
            size=size,
            content=content,
        )
    except ValidationError as exc:
        status_code = 413 if exc.code == "file_too_large" else 422
        return _error_response(status_code, exc.code, str(exc))
    except IngestionError as exc:
        if exc.code in ("unsupported_file_type", "empty_content", "malformed_file"):
            status_code = 422
        else:
            status_code = 500
        return _error_response(status_code, exc.code, str(exc))
    except Exception:
        logger.exception("Unexpected error during resume extraction")
        return _error_response(
            500, "extraction_failed", "An unexpected error occurred."
        )

    return JSONResponse(status_code=200, content=result.model_dump())


@router.post("/parse")
async def parse_resume_endpoint(
    file: Annotated[
        UploadFile,
        File(description="Resume file (PDF, DOCX, or TXT)"),
    ],
) -> JSONResponse:
    content = await file.read()
    size = len(content)

    try:
        result: ExtractionResult = extract_resume_text(
            filename=file.filename,
            content_type=file.content_type,
            size=size,
            content=content,
        )
        resume = parse_resume(result.extracted_text, file_type=result.file_type)
    except ValidationError as exc:
        status_code = 413 if exc.code == "file_too_large" else 422
        return _error_response(status_code, exc.code, str(exc))
    except IngestionError as exc:
        if exc.code in ("unsupported_file_type", "empty_content", "malformed_file"):
            status_code = 422
        else:
            status_code = 500
        return _error_response(status_code, exc.code, str(exc))
    except ParsingError as exc:
        return _error_response(422, exc.code, str(exc))
    except Exception:
        logger.exception("Unexpected error during resume parsing")
        return _error_response(500, "parsing_failed", "An unexpected error occurred.")

    return JSONResponse(status_code=200, content=resume.model_dump())


class ATSRequestBody(BaseModel):
    """Transient request: a fully parsed Resume, analysed without a JD."""

    resume: Resume


@router.post("/ats-analysis")
async def ats_analysis(body: ATSRequestBody) -> JSONResponse:
    """ATS readiness / resume-quality analysis (deterministic, transient).

    Evaluates an already-parsed Resume WITHOUT a job description using
    deterministic, explainable heuristics. No persistence, no files, no
    external calls, no embeddings, no LLM: the request and result exist only
    for the lifetime of this call and nothing is logged.
    """
    try:
        result = analyze_resume(body.resume)
    except Exception:
        logger.exception("Unexpected error during ATS analysis")
        return _error_response(
            500,
            "ats_analysis_failed",
            "An unexpected error occurred.",
        )
    return JSONResponse(status_code=200, content=result.model_dump(mode="json"))


class JobSpecificATSRequestBody(BaseModel):
    """Transient request: a parsed Resume plus a parsed JobDescription."""

    resume: Resume
    job_description: JobDescription


@router.post("/job-specific-ats")
async def job_specific_ats(body: JobSpecificATSRequestBody) -> JSONResponse:
    """Job-Specific ATS Coverage analysis (deterministic, transient).

    Measures how completely the job description's explicit terminology is
    represented in the resume. No persistence, no files, no external calls,
    no embeddings, no LLM. The request and result exist only for the lifetime
    of this call and nothing is logged.
    """
    try:
        result = analyze_job_specific_ats(body.resume, body.job_description)
    except Exception:
        logger.exception("Unexpected error during job-specific ATS analysis")
        return _error_response(
            500,
            "job_specific_ats_failed",
            "An unexpected error occurred.",
        )
    return JSONResponse(status_code=200, content=result.model_dump(mode="json"))
