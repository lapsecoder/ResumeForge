"""Data-quality diagnostics (PII-presence, emptiness, type consistency).

Only aggregate counts are reported. Contact-like values (email/phone shapes)
are detected with regexes and counted, never surfaced.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from ml.audit._helpers import flatten_value

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?:\+?\d[\d .-]{8,}\d)")

_SCAN_COLUMNS = {
    "resumes": [
        "role", "seniority", "industry", "education", "summary",
        "experience_bullets",
    ],
    "jobs": [
        "job_title", "seniority", "industry", "description",
        "responsibilities",
    ],
}


def _pii_scan(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    matches = {"email": 0, "phone": 0}
    for column in columns:
        if column not in frame.columns:
            continue
        for value in frame[column]:
            for text in flatten_value(value):
                if _EMAIL.search(str(text)):
                    matches["email"] += 1
                if _PHONE.search(str(text)):
                    matches["phone"] += 1
    return matches


def quality_diagnose(
    resumes: pd.DataFrame, jobs: pd.DataFrame, matches: pd.DataFrame
) -> dict[str, Any]:
    resume_pii = _pii_scan(resumes, _SCAN_COLUMNS["resumes"])
    job_pii = _pii_scan(jobs, _SCAN_COLUMNS["jobs"])
    return {
        "resumes": {
            "count": int(len(resumes)),
            "empty_role": int(resumes["role"].fillna("").eq("").sum()),
            "empty_summary": int(resumes["summary"].fillna("").eq("").sum()),
            "empty_bullets": int(resumes["experience_bullets"].map(len).eq(0).sum()),
            "empty_skills": int(resumes["skills"].map(len).eq(0).sum()),
            "negative_years_experience": int(
                (resumes["years_experience"] < 0).sum()
            ),
            "max_skill_list_len": int(resumes["skills"].map(len).max()),
            "detected_email_like": int(resume_pii["email"]),
            "detected_phone_like": int(resume_pii["phone"]),
        },
        "jobs": {
            "count": int(len(jobs)),
            "empty_title": int(jobs["job_title"].fillna("").eq("").sum()),
            "empty_description": int(jobs["description"].fillna("").eq("").sum()),
            "empty_must_have": int(jobs["must_have_skills"].map(len).eq(0).sum()),
            "empty_nice_to_have": int(
                jobs["nice_to_have_skills"].map(len).eq(0).sum()
            ),
            "detected_email_like": int(job_pii["email"]),
            "detected_phone_like": int(job_pii["phone"]),
        },
        "matches": {
            "rows": int(len(matches)),
            "duplicate_job_ids": int(matches["job_id"].duplicated().sum()),
            "rows_referencing_missing_job": int(
                (~matches["job_id"].isin(set(jobs["job_id"]))).sum()
            ),
            "pairs_referencing_missing_resume": int(
                (~matches["relevant_resume_ids"].explode().isin(set(resumes["resume_id"])))
                .sum()
            ),
        },
    }
