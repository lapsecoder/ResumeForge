"""Deterministic job-description parsing package.

Everything here is transient and rule-based. Job descriptions are never
persisted, never logged, and no external service or LLM is involved.
"""

from app.job_parsing.parser import ParsingError, parse_job_description
from app.job_parsing.schemas import JobDescription

__all__ = ["JobDescription", "ParsingError", "parse_job_description"]
