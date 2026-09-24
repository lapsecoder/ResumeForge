"""Job-description endpoints: deterministic parsing.

Transient and privacy-first: upload → validate → extract → normalise → parse →
return. The parsed model exists only for the lifetime of the request, exactly
like the resume pipeline.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from app.ingestion.schemas import ExtractionResult
from app.ingestion.service import IngestionError, extract_resume_text
from app.ingestion.validators import ValidationError
from app.job_parsing.parser import ParsingError, parse_job_description

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


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


@router.post("/parse")
async def parse_job_description_endpoint(
    file: Annotated[
        UploadFile,
        File(description="Job description file (PDF, DOCX, or TXT)"),
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
        job = parse_job_description(result.extracted_text, file_type=result.file_type)
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
        logger.exception("Unexpected error during job-description parsing")
        return _error_response(500, "parsing_failed", "An unexpected error occurred.")

    return JSONResponse(status_code=200, content=job.model_dump())
