"""Adapter: audited dataset rows → Resume / JobDescription.

Only genuinely available fields are mapped.  Every field that does NOT exist in
the synthetic dataset is deliberately left absent.  No experience dates,
institutions, certifications, or qualifications are fabricated.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from app.job_parsing.schemas import JobDescription, JobMetadata
from app.matching.service import match_resume_to_job
from app.parsing.schemas import (
    ConfidenceLevel,
    Resume,
    ResumeMetadata,
    SkillSet,
)

_REFERENCE_DATE = date(2026, 1, 1)


def _to_str_list(value: Any) -> list[str]:
    """Normalise a dataset cell to a list of strings.

    Handles np.ndarray cells, lists, tuples, NaN, None, and scalars without
    raising on ambiguous truth values.
    """
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(x) for x in value.tolist() if x not in (None, "", np.nan)]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if x not in (None, "")]
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(value)]


def _word_count(*parts: list[str] | str | None) -> int:
    tokens: list[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, list):
            for token in part:
                tokens.extend(str(token).split())
        else:
            tokens.extend(str(part).split())
    return len(tokens)


def resume_from_row(row: pd.Series) -> Resume:
    """Map a dataset resume row to a Resume. No fields are fabricated."""
    skill_list = _to_str_list(row.get("skills"))
    summary = str(row.get("summary") or "")
    bullets = _to_str_list(row.get("experience_bullets"))

    wc = _word_count(summary, bullets, skill_list)

    return Resume(
        summary=summary,
        experience=[],
        education=[],
        skills=SkillSet(all=skill_list),
        projects=[],
        certifications=[],
        custom_sections=[],
        metadata=ResumeMetadata(
            word_count=wc,
            file_type="txt",
            overall_confidence=ConfidenceLevel.HIGH,
        ),
    )


def job_from_row(row: pd.Series) -> JobDescription:
    """Map a dataset job row to a JobDescription. No fields are fabricated."""
    required = _to_str_list(row.get("must_have_skills"))
    preferred = _to_str_list(row.get("nice_to_have_skills"))
    responsibilities = _to_str_list(row.get("responsibilities"))
    desc = str(row.get("description") or "")

    wc = _word_count(desc, responsibilities, required, preferred)

    return JobDescription(
        title=str(row.get("job_title") or ""),
        summary=desc,
        responsibilities=responsibilities,
        required_skills=required,
        preferred_skills=preferred,
        metadata=JobMetadata(
            word_count=wc,
            overall_confidence=ConfidenceLevel.HIGH,
        ),
    )


def compute_phase5a_scores(
    pairs: pd.DataFrame,
    resumes_lookup: dict[str, pd.Series],
    jobs_lookup: dict[str, pd.Series],
) -> list[float | None]:
    """Compute the Phase-5A overall_score for every pair.

    Returns a list aligned with ``pairs.index``.  ``None`` scores (no
    evaluable component) are converted to ``0.0`` for downstream
    thresholding; the fraction of None scores is recorded in the result
    metadata so the adapter's coverage is transparent.

    Conducts one mapping per unique resume/job (not per pair), then scores
    each pair with the pre-built objects.
    """
    resume_cache: dict[str, Resume] = {
        rid: resume_from_row(row)
        for rid, row in resumes_lookup.items()
    }
    job_cache: dict[str, JobDescription] = {
        jid: job_from_row(row)
        for jid, row in jobs_lookup.items()
    }

    scores: list[float | None] = []
    for _, row in pairs.iterrows():
        rid = str(row["resume_id"])
        jid = str(row["job_id"])
        resume = resume_cache[rid]
        job = job_cache[jid]
        result = match_resume_to_job(resume, job, reference_date=_REFERENCE_DATE)
        scores.append(result.overall_score)
    return scores


def score_to_label(
    scores: list[float | None],
    threshold: float,
) -> list[int]:
    """Convert continuous Phase-5A scores to binary labels.

    None scores are predicted as 0 (negative).  The fraction of None
    predictions is recorded separately in the evaluation metadata.
    """
    return [1 if (s is not None and s >= threshold) else 0 for s in scores]


def derive_threshold_on_train(
    scores: list[float | None],
    y_train: list[int],
) -> tuple[float, dict[str, float]]:
    """Grid-search for the threshold maximizing macro-F1 on training data.

    Returns (frozen_threshold, {candidate_threshold: macro_f1}).
    """
    from ml.metrics import macro_f1 as _mf1

    candidates = [round(t * 0.1, 1) for t in range(0, 1001)]
    best_thresh = 0.0
    best_f1 = -1.0
    threshold_curve: dict[str, float] = {}
    for t in candidates:
        preds = [1 if (s is not None and s >= t) else 0 for s in scores]
        f1 = _mf1(y_train, preds)
        threshold_curve[str(t)] = round(f1, 4)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, threshold_curve
