"""Deterministic feature builders for resume/job pairs.

Builds keyword overlap and other deterministic features from text fields.
These are stateless transformers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.preprocessing.text import tokenize


def _safe_is_na(val: object) -> bool:
    try:
        result = pd.isna(val)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
        return False
    except (ValueError, TypeError):
        return False


def _skill_set(skills_val: object) -> set[str]:
    if _safe_is_na(skills_val):
        return set()
    if isinstance(skills_val, (list, tuple, np.ndarray)):
        return {str(s).lower().strip() for s in skills_val if not _safe_is_na(s)}
    return {s.lower().strip() for s in str(skills_val).split(",") if s.strip()}


def _keyword_overlap(resume_skills: set[str], must_have: set[str]) -> float:
    if not must_have:
        return 0.0
    overlap = resume_skills & must_have
    return len(overlap) / len(must_have) if must_have else 0.0


def build_deterministic_match_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic match features (keyword overlap, etc.)."""
    rows = []
    for _, row in pairs.iterrows():
        resume_skills = _skill_set(row.get("resume_skills"))
        must_have = _skill_set(row.get("job_must_have_skills"))
        nice_have = _skill_set(row.get("job_nice_to_have_skills"))

        required_kw = _keyword_overlap(resume_skills, must_have)
        preferred_kw = _keyword_overlap(resume_skills, nice_have)
        combined_kw = _keyword_overlap(resume_skills, must_have | nice_have)

        resume_text_tokens = set()
        for col in ("summary", "experience_bullets"):
            try:
                val = row[col]
                if not _safe_is_na(val):
                    if isinstance(val, (list, tuple)):
                        val = " ".join(str(v) for v in val)
                    resume_text_tokens.update(tokenize(str(val)))
            except (KeyError, ValueError):
                pass

        job_text_tokens = set()
        for col in ("description", "responsibilities", "requirements"):
            try:
                val = row[col]
                if not _safe_is_na(val):
                    if isinstance(val, (list, tuple)):
                        val = " ".join(str(v) for v in val)
                    job_text_tokens.update(tokenize(str(val)))
            except (KeyError, ValueError):
                pass

        if resume_text_tokens and job_text_tokens:
            overlap_tokens = resume_text_tokens & job_text_tokens
            text_cosine_sim = len(overlap_tokens) / (
                (len(resume_text_tokens) * len(job_text_tokens)) ** 0.5 + 1e-10
            )
        else:
            text_cosine_sim = 0.0

        hybrid = 0.5 * required_kw + 0.3 * text_cosine_sim + 0.2 * preferred_kw

        rows.append({
            "keyword_overlap_required": required_kw,
            "keyword_overlap_preferred": preferred_kw,
            "keyword_overlap_combined": combined_kw,
            "text_token_overlap": text_cosine_sim,
            "hybrid_score": hybrid,
        })

    return pd.DataFrame(rows, index=pairs.index, dtype=float)
