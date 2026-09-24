"""Synthetic-content leakage diagnostics.

Both metadata leakage (identical template phrases, repeated boilerplate) and
the shared vocabulary between resumes and jobs are quantified as aggregate
counts and percentages. No sample text is ever returned.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd

from ml.audit._helpers import flatten_value

_SUMMARY_TEMPLATE = re.compile(
    r"^(.*?) with (\d+) years? of experience in (.*?)$", re.IGNORECASE
)
_WORD = re.compile(r"[a-z0-9']+", re.IGNORECASE)


def _words(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD.finditer(text)}


def _bullet_imports(resumes: pd.DataFrame) -> tuple[int, int, int, int]:
    bullet_counter: Counter[str] = Counter()
    intra_duplicates = 0
    for _, row in resumes.iterrows():
        bullets = flatten_value(row.get("experience_bullets"))
        bullet_counter.update(str(b) for b in bullets)
        seen: set[str] = set()
        for bullet in bullets:
            if str(bullet) in seen:
                intra_duplicates += 1
            seen.add(str(bullet))
    n_total = sum(bullet_counter.values())
    n_distinct = len(bullet_counter)
    shared_occurrences = sum(c for c in bullet_counter.values() if c > 1)
    return n_total, n_distinct, n_total - shared_occurrences, intra_duplicates


def _summary_template_counts(resumes: pd.DataFrame) -> tuple[int, int]:
    total = len(resumes)
    matched = 0
    for value in resumes["summary"].fillna(""):
        if _SUMMARY_TEMPLATE.match(str(value)):
            matched += 1
    return total, matched


def leak_resumes(resumes: pd.DataFrame) -> dict[str, Any]:
    total, template = _summary_template_counts(resumes)
    (
        bullet_total,
        bullet_distinct,
        bullet_unique_occ,
        bullet_intra,
    ) = _bullet_imports(resumes)
    return {
        "resume_count": int(total),
        "summary_follows_template": int(template),
        "pct_summary_follows_template": float(template / total * 100.0),
        "experience_bullets": {
            "total": int(bullet_total),
            "distinct_strings": int(bullet_distinct),
            "occurrences_of_shared_strings": int(bullet_total - bullet_unique_occ),
            "pct_occurrences_of_shared_strings": float(
                (bullet_total - bullet_unique_occ) / bullet_total * 100.0
            ),
            "intra_resume_duplicated_bullets": int(bullet_intra),
        },
    }


def leak_jobs(jobs: pd.DataFrame) -> dict[str, Any]:
    return {
        "job_count": int(len(jobs)),
        "unique_job_titles": int(jobs["job_title"].nunique()),
        "unique_descriptions": int(jobs["description"].nunique()),
        "unique_responsibility_blocks": int(
            jobs["responsibilities"].astype(str).nunique()
        ),
        "pct_unique_descriptions": float(
            jobs["description"].nunique() / len(jobs) * 100.0
        ),
    }


def vocab_overlap(resumes: pd.DataFrame, jobs: pd.DataFrame) -> dict[str, Any]:
    resume_words: Counter[str] = Counter()
    for value in resumes["role"].dropna():
        resume_words.update(_words(str(value)))
    for value in resumes["summary"].dropna():
        resume_words.update(_words(str(value)))
    for value in resumes["experience_bullets"]:
        for bullet in flatten_value(value):
            resume_words.update(_words(str(bullet)))

    job_words: Counter[str] = Counter()
    for value in jobs["job_title"].dropna():
        job_words.update(_words(str(value)))
    for value in jobs["description"].dropna():
        job_words.update(_words(str(value)))
    for value in jobs["responsibilities"]:
        for bullet in flatten_value(value):
            job_words.update(_words(str(bullet)))

    resume_skills = {str(v) for vv in resumes["skills"] for v in flatten_value(vv)}
    job_must = {str(v) for vv in jobs["must_have_skills"] for v in flatten_value(vv)}

    return {
        "resume_words": int(sum(resume_words.values())),
        "job_words": int(sum(job_words.values())),
        "shared_narrative_words_fraction": float(
            len(set(resume_words) & set(job_words))
            / max(1, len(set(resume_words) | set(job_words)))
            * 100.0
        ),
        "resume_skill_vocab": int(len(resume_skills)),
        "job_must_have_vocab": int(len(job_must)),
        "skills_shared_fraction": float(
            len(resume_skills & job_must) / max(1, len(job_must)) * 100.0
        ),
    }
