"""General ATS readiness / resume-quality analysis (Phase 6A).

Deterministic, explainable, job-description-free analysis of a parsed resume.
Everything is transient: no persistence, no files, no external calls, no LLM,
no embeddings, and no logging of resume contents.
"""

from app.ats_analysis.schemas import (
    AnalysisMetadata,
    ATSReadinessResult,
    CategoryScore,
    Finding,
    Severity,
)
from app.ats_analysis.service import analyze_resume

__all__ = [
    "ATSReadinessResult",
    "AnalysisMetadata",
    "CategoryScore",
    "Finding",
    "Severity",
    "analyze_resume",
]
