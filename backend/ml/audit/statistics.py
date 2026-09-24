"""Aggregate statistics over resumes and jobs.

Nothing in this module ever emits raw resume/job text or PII - only counts,
distributions and summaries.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from ml.audit._helpers import flatten_value


def _skill_counts(frame: pd.DataFrame, columns: list[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for column in columns:
        if column not in frame.columns:
            continue
        for value in frame[column]:
            for skill in flatten_value(value):
                counter[str(skill)] += 1
    return counter


def summarize_resumes(resumes: pd.DataFrame) -> dict[str, Any]:
    n = len(resumes)
    skills = _skill_counts(resumes, ["skills"])
    nullable_cols = ["role", "seniority", "industry", "education", "summary"]
    return {
        "count": int(n),
        "columns": list(resumes.columns),
        "nulls": {
            col: int(resumes[col].isna().sum())
            for col in nullable_cols if col in resumes
        },
        "nulls_non_optional": {
            col: int(resumes[col].isna().sum())
            for col in ["resume_id", "role", "seniority", "years_experience", "skills"]
            if col in resumes
        },
        "duplicate_resume_ids": int(
            resumes["resume_id"].duplicated().sum()
            if "resume_id" in resumes
            else -1
        ),
        "seniority_counts": {
            str(k): int(v)
            for k, v in (resumes["seniority"].fillna("").value_counts().items())
        },
        "industry_counts": {
            str(k): int(v)
            for k, v in (resumes["industry"].fillna("").value_counts().items())
        },
        "role_counts_top": {
            str(k): int(v)
            for k, v in (resumes["role"].fillna("").value_counts().head(30).items())
        },
        "years_experience": {
            "mean": float(resumes["years_experience"].mean()),
            "median": float(resumes["years_experience"].median()),
            "min": int(resumes["years_experience"].min()),
            "max": int(resumes["years_experience"].max()),
            "by_seniority_mean": {
                str(k): float(v)
                for k, v in (
                    resumes.groupby("seniority")["years_experience"].mean().items()
                )
            },
        },
        "education_counts": {
            str(k): int(v)
            for k, v in (resumes["education"].fillna("").value_counts().items())
        },
        "skills_per_resume": {
            "mean": float(resumes["skills"].map(len).mean()),
            "min": int(resumes["skills"].map(len).min()),
            "max": int(resumes["skills"].map(len).max()),
        },
        "bullets_per_resume": {
            "mean": float(resumes["experience_bullets"].map(len).mean()),
            "min": int(resumes["experience_bullets"].map(len).min()),
            "max": int(resumes["experience_bullets"].map(len).max()),
        },
        "top_20_skills": list(skills.items()),
    }


def summarize_jobs(jobs: pd.DataFrame) -> dict[str, Any]:
    must_have = _skill_counts(jobs, ["must_have_skills"])
    nice_to_have = _skill_counts(jobs, ["nice_to_have_skills"])
    return {
        "count": int(len(jobs)),
        "columns": list(jobs.columns),
        "nulls": {
            col: int(jobs[col].isna().sum())
            for col in ["job_id", "job_title", "seniority", "industry", "description"]
            if col in jobs
        },
        "seniority_counts": {
            str(k): int(v)
            for k, v in (jobs["seniority"].fillna("").value_counts().items())
        },
        "industry_counts": {
            str(k): int(v)
            for k, v in (jobs["industry"].fillna("").value_counts().items())
        },
        "title_counts_top": {
            str(k): int(v)
            for k, v in (jobs["job_title"].fillna("").value_counts().head(30).items())
        },
        "must_have_per_job": {
            "mean": float(jobs["must_have_skills"].map(len).mean()),
            "min": int(jobs["must_have_skills"].map(len).min()),
            "max": int(jobs["must_have_skills"].map(len).max()),
        },
        "responsibilities_per_job": {
            "mean": float(jobs["responsibilities"].map(len).mean()),
            "min": int(jobs["responsibilities"].map(len).min()),
            "max": int(jobs["responsibilities"].map(len).max()),
        },
        "description_length_chars": {
            "mean": float(jobs["description"].str.len().mean()),
            "min": int(jobs["description"].str.len().min()),
            "max": int(jobs["description"].str.len().max()),
        },
        "top_20_must_have": list(must_have.items()),
        "top_20_nice_to_have": list(nice_to_have.items()),
    }


def cross_section(resumes: pd.DataFrame, jobs: pd.DataFrame) -> dict[str, Any]:
    resume_roles = set(resumes["role"].fillna(""))
    job_roles = set(jobs["job_title"].fillna(""))
    return {
        "resume_roles_count": int(len(resume_roles)),
        "job_titles_count": int(len(job_roles)),
        "shared_role_labels": int(len(resume_roles & job_roles)),
        "resume_industries": int(resumes["industry"].nunique()),
        "job_industries": int(jobs["industry"].nunique()),
        "shared_industries": int(
            len(
                set(resumes["industry"].dropna())
                & set(jobs["industry"].dropna())
            )
        ),
        "seniority_scale_resumes": {
            str(k): int(v)
            for k, v in (resumes["seniority"].fillna("").value_counts().items())
        },
        "seniority_scale_jobs": {
            str(k): int(v)
            for k, v in (jobs["seniority"].fillna("").value_counts().items())
        },
    }
