"""ResumeForge backend — FastAPI application entry point."""

import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.copilot import router as copilot_router
from app.api.v1.jobs import router as jobs_router
from app.api.v1.matching import router as matching_router
from app.api.v1.resumes import router as resumes_router
from app.api.v1.roles import router as roles_router
from app.core.config import settings

app = FastAPI(
    title="ResumeForge API",
    version="0.1.0",
    description="Backend for the ResumeForge resume and career platform.",
)

_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(resumes_router)
app.include_router(jobs_router)
app.include_router(matching_router)
app.include_router(roles_router)
app.include_router(copilot_router)


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Structured 422 that never echoes the offending input values (PII-safe)."""
    details = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", []))
        details.append(f"{loc}: {error.get('type', 'invalid')}")
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "; ".join(details),
                "request_id": str(uuid.uuid4()),
            }
        },
    )


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}
