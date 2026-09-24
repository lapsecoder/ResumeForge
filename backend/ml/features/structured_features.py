"""Structured feature builders for resume/job pairs.

Builds deterministic features from structured fields.  No learned state —
these are stateless transformers.
"""

from __future__ import annotations

import pandas as pd

from ml.preprocessing.structured import (
    degree_level,
    matches_industry,
    matches_seniority,
    role_overlap,
    seniority_tier,
)


def _cell(row: pd.Series, column: str) -> str:
    val = row.get(column)
    return str(val) if pd.notna(val) else ""


def build_structured_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """Build structured features from pair rows that have resume/job columns."""
    rows = []
    for _, row in pairs.iterrows():
        features: dict[str, float] = {}

        resume_seniority = _cell(row, "resume_seniority")
        job_seniority = _cell(row, "job_seniority")
        resume_industry = _cell(row, "resume_industry")
        job_industry = _cell(row, "job_industry")
        resume_role = _cell(row, "resume_role")
        job_title = _cell(row, "job_title")
        education = _cell(row, "resume_education")

        years_raw = row.get("resume_years_experience")
        if pd.notna(years_raw):
            try:
                years = float(years_raw)
            except (TypeError, ValueError):
                years = 0.0
        else:
            years = 0.0

        features["years_experience"] = years
        features["years_experience_bucket"] = {
            "junior": 0.0, "mid": 1.0, "senior": 2.0
        }.get(seniority_tier(years) if years >= 0 else "junior", 0.0)

        deg = degree_level(education)
        for level in ("phd", "masters", "bachelors", "diploma", "high_school"):
            features[f"degree_{level}"] = 1.0 if deg == level else 0.0
        features["degree_unknown"] = 1.0 if deg is None else 0.0

        features["same_seniority"] = (
            1.0 if matches_seniority(resume_seniority, job_seniority) else 0.0
        )
        features["same_industry"] = (
            1.0 if matches_industry(resume_industry, job_industry) else 0.0
        )
        features["same_role"] = 1.0 if role_overlap(resume_role, job_title) else 0.0

        rows.append(features)

    return pd.DataFrame(rows, index=pairs.index)


def structured_feature_names() -> list[str]:
    """Return the ordered list of structured feature column names."""
    names = ["years_experience", "years_experience_bucket"]
    for level in ("phd", "masters", "bachelors", "diploma", "high_school"):
        names.append(f"degree_{level}")
    names.append("degree_unknown")
    names.extend(["same_seniority", "same_industry", "same_role"])
    return names
