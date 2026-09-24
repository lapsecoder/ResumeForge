"""Target reconstruction and derivability analysis.

The dataset carries no explicit binary label: ``matches`` maps every job_id to
a list of "relevant" resume_ids. This module quantifies how faithfully the
published generation rule (must-have coverage >= 60%, random sample capped at
30) recovers the published labels, so later supervised work can reason about
label noise.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from ml.audit._helpers import flatten_value

MUST_HAVE_THRESHOLD = 0.6
MAX_RELEVANT_PER_JOB = 30


def summarize_matches(matches: pd.DataFrame) -> dict[str, Any]:
    relevant = matches["relevant_resume_ids"].map(flatten_value)
    counts = relevant.map(len)
    return {
        "rows": int(len(matches)),
        "unique_jobs": int(matches["job_id"].nunique()),
        "per_job_relevant_count": {
            "mean": float(counts.mean()),
            "min": int(counts.min()),
            "max": int(counts.max()),
            "distinct_values": sorted(int(x) for x in counts.unique()),
        },
    }


def _skill_index(resumes: pd.DataFrame) -> dict[str, set[str]]:
    inverted: dict[str, set[str]] = {}
    for _, row in resumes.iterrows():
        resume_id = str(row["resume_id"])
        for skill in flatten_value(row.get("skills")):
            inverted.setdefault(str(skill), set()).add(resume_id)
    return inverted


def derivability(
    resumes: pd.DataFrame, jobs: pd.DataFrame, matches: pd.DataFrame
) -> dict[str, Any]:
    """Compare published labels with the threshold-reachable candidate set.

    For each job we rebuild the notebook's candidate filter: a resume is a
    candidate when it contains at least ``ceil(0.6 * |must|)`` of the job's
    must-have skills (counted as distinct matching vocabulary items, exactly
    like the generator). Aggregates only - no candidacy decisions are stored.
    """
    inverted = _skill_index(resumes)
    relevant_by_job = {
        str(row["job_id"]): set(flatten_value(row["relevant_resume_ids"]))
        for _, row in matches.iterrows()
    }

    total_jobs = len(jobs)
    capped_jobs = 0
    candidate_pairs = 0
    candidate_cap_excess = 0
    relevant_pairs = 0
    overlap_pairs = 0
    empty_must_jobs = 0
    for _, row in jobs.iterrows():
        job_id = str(row["job_id"])
        must_have = [str(s) for s in flatten_value(row.get("must_have_skills"))]
        if not must_have:
            empty_must_jobs += 1
            continue
        need = int((MUST_HAVE_THRESHOLD * len(must_have)) + 0.9999)
        hits: Counter[str] = Counter()
        for skill in must_have:
            for resume_id in inverted.get(skill, ()):
                hits[resume_id] += 1
        candidate = {resume_id for resume_id, count in hits.items() if count >= need}

        relevant = relevant_by_job.get(job_id, set())
        candidate_pairs += len(candidate)
        relevant_pairs += len(relevant)
        overlap_pairs += len(relevant & candidate)
        if len(candidate) > MAX_RELEVANT_PER_JOB:
            capped_jobs += 1
        candidate_cap_excess += max(0, len(candidate) - MAX_RELEVANT_PER_JOB)

    return {
        "threshold_must_have": MUST_HAVE_THRESHOLD,
        "cap": MAX_RELEVANT_PER_JOB,
        "jobs": int(total_jobs),
        "jobs_with_empty_must_have": int(empty_must_jobs),
        "jobs_with_more_candidates_than_cap": int(capped_jobs),
        "published_relevant_pairs": int(relevant_pairs),
        "rule_candidate_pairs": int(candidate_pairs),
        "overlap_relevant_and_candidate": int(overlap_pairs),
        "pct_relevant_published_within_candidate": float(
            (overlap_pairs / relevant_pairs * 100.0) if relevant_pairs else 0.0
        ),
        "fraction_candidate_pairs_discarded_by_cap": float(
            (candidate_cap_excess / candidate_pairs * 100.0) if candidate_pairs else 0.0
        ),
    }


def resume_exposure(matches: pd.DataFrame) -> dict[str, Any]:
    """How often each resume_id appears as relevant (label coverage check)."""
    exposure: Counter[str] = Counter()
    for _, row in matches.iterrows():
        for resume_id in flatten_value(row["relevant_resume_ids"]):
            exposure[str(resume_id)] += 1
    values = list(exposure.values())
    return {
        "distinct_resumes_used": int(len(values)),
        "counts_per_resume": {
            "mean": float(sum(values) / len(values)) if values else 0.0,
            "min": int(min(values)) if values else 0,
            "max": int(max(values)) if values else 0,
        },
        "histogram_buckets": {
            f"{lo}-{hi}": sum(1 for v in values if lo <= v < hi)
            for lo, hi in (
                (1, 2), (2, 5), (5, 10), (10, 20),
                (20, MAX_RELEVANT_PER_JOB + 1),
            )
        },
    }
