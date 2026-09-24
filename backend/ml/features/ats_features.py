"""ATS feature builders (Phase 6A/6B derived, non-quasi-generative subset).

Builds general ATS quality heuristics and the allowed Phase 6B sub-fields.
Excludes ``required_coverage`` and ``overall_coverage`` per Phase 6D.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_ACTION_VERBS = {
    "achieved", "improved", "reduced", "increased", "developed", "implemented",
    "designed", "led", "managed", "created", "delivered", "optimized",
    "streamlined", "automated", "established", "launched", "scaled",
    "negotiated", "resolved", "organized", "coordinated", "supervised",
}

_QUANTITATIVE_RE = re.compile(r"\b\d[\d,.]*%?\b")


def _count_action_verbs(text: str) -> int:
    tokens = text.lower().split()
    return sum(1 for t in tokens if t.strip(".,;:") in _ACTION_VERBS)


def _count_quantitative(text: str) -> int:
    return len(_QUANTITATIVE_RE.findall(text))


def _bullet_count(text: str) -> int:
    return sum(
        1
        for line in text.split("\n")
        if line.strip().startswith("-") or line.strip().startswith("*")
    )


def _safe_is_na(val: object) -> bool:
    try:
        result = pd.isna(val)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
        return False
    except (ValueError, TypeError):
        return False


def _section_completeness(row: pd.Series) -> float:
    sections = ["summary", "experience_bullets", "skills", "education"]
    present = 0
    for s in sections:
        try:
            val = row[s]
            if not _safe_is_na(val) and str(val).strip():
                present += 1
        except (KeyError, ValueError):
            pass
    return present / len(sections)


def build_ats_6a_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """Build general ATS quality features (Phase 6A style)."""
    rows = []
    for _, row in pairs.iterrows():
        resume_text_parts: list[str] = []
        for col in ("summary", "experience_bullets", "skills"):
            try:
                val = row[col]
                if not _safe_is_na(val):
                    if isinstance(val, (list, tuple)):
                        resume_text_parts.append(" ".join(str(v) for v in val))
                    else:
                        resume_text_parts.append(str(val))
            except (KeyError, ValueError):
                pass
        resume_text = " ".join(resume_text_parts)

        rows.append({
            "action_verb_count": float(_count_action_verbs(resume_text)),
            "quantitative_count": float(_count_quantitative(resume_text)),
            "section_completeness": _section_completeness(row),
        })

    return pd.DataFrame(rows, index=pairs.index, dtype=float)


def _skill_token_set(skills_val: object) -> set[str]:
    if _safe_is_na(skills_val):
        return set()
    if isinstance(skills_val, (list, tuple, np.ndarray)):
        return {str(s).lower().strip() for s in skills_val if not _safe_is_na(s)}
    return {s.lower().strip() for s in str(skills_val).split(",") if s.strip()}


def build_ats_6b_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """Build job-specific ATS features (allowed sub-fields only).

    Excludes ``required_coverage`` and ``overall_coverage`` per Phase 6D.
    """
    rows = []
    for _, row in pairs.iterrows():
        resume_skills = _skill_token_set(row.get("resume_skills"))
        nice_have = _skill_token_set(row.get("job_nice_to_have_skills"))
        must_have = _skill_token_set(row.get("job_must_have_skills"))

        preferred_coverage = (
            len(resume_skills & nice_have) / len(nice_have) if nice_have else 0.0
        )

        resume_text_parts: list[str] = []
        for col in ("summary", "experience_bullets"):
            try:
                val = row[col]
                if not _safe_is_na(val):
                    if isinstance(val, (list, tuple)):
                        resume_text_parts.append(" ".join(str(v) for v in val))
                    else:
                        resume_text_parts.append(str(val))
            except (KeyError, ValueError):
                pass
        resume_text_tokens = set(" ".join(resume_text_parts).lower().split())

        req_evidence = (
            len(resume_text_tokens & must_have) / len(must_have)
            if must_have else 0.0
        )
        pref_evidence = (
            len(resume_text_tokens & nice_have) / len(nice_have)
            if nice_have else 0.0
        )

        rows.append({
            "preferred_coverage": preferred_coverage,
            "evidence_supported_required": req_evidence,
            "evidence_supported_preferred": pref_evidence,
        })

    return pd.DataFrame(rows, index=pairs.index, dtype=float)
