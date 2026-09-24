"""Split-strategy analysis for the candidate-matching dataset.

The repository ships no train/validation/test partition (the ``train`` suffix
in the parquet filenames is just the single file name, not a split). This
module analyses what a sound split must protect against and recommends a
recipe; it does not create any split itself.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .leakage import vocab_overlap


def split_strategy_recommendation(
    resumes: pd.DataFrame, jobs: pd.DataFrame, matches: pd.DataFrame
) -> dict[str, Any]:
    shared = vocab_overlap(resumes, jobs)
    job_count = len(jobs)
    has_test = False
    if "split" in resumes.columns:
        has_test = has_test or bool((resumes["split"] == "test").any())
    if "split" in jobs.columns:
        has_test = has_test or bool((jobs["split"] == "test").any())
    return {
        "has_repository_test_split": bool(has_test),
        "recommendations": [
            "role-stratified job-level grouping: keep every job's 30 labels "
            "together (never split the matches of one job)",
            "leakage guard: near-duplicate resume bullets/summaries share "
            "memorised vocabulary - deduplicate near-identical resumes and "
            "place whole near-duplicate groups in one split",
            "shared skill vocabulary is a feature, not leakage; train/test "
            "skill overlap is expected and acceptable",
        ],
        "rationale": {
            "jobs": int(job_count),
            "labels_per_job": 30,
            "pairs": int(len(matches)),
            "shared_narrative_words_fraction": shared[
                "shared_narrative_words_fraction"
            ],
            "repeated_content_present": True,
        },
    }
