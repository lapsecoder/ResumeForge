"""Deterministic resume parsing: normalised text → structured Resume.

All parsing is local, deterministic, and transient. No data is persisted and
no resume content is logged.
"""

from app.parsing.parser import parse_resume
from app.parsing.schemas import Resume

__all__ = ["Resume", "parse_resume"]
