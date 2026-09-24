"""Error analysis helpers for Phase 6F.

Operates on test-split predictions: TP/TN/FP/FN categorisation plus
deterministic, consenting representative-example summaries.  Summaries never
drain raw resume/job text — they emit compact derived signals (role, seniority,
overlap fractions, cosine) so the report stays privacy-preserving and readable.
"""

from __future__ import annotations

import random
from typing import Any, Callable

import pandas as pd

from ml.metrics import (
    confusion_counts,
    macro_f1,
    per_class_report,
)


def categorize(
    y_true: list[int], y_pred: list[int]
) -> dict[str, list[int]]:
    """Return index lists belonging to TP/TN/FP/FN for the binary split."""
    buckets: dict[str, list[int]] = {
        "tp": [], "tn": [], "fp": [], "fn": [],
    }
    for i, (t, p) in enumerate(zip(y_true, y_pred, strict=True)):
        if t == 1 and p == 1:
            buckets["tp"].append(i)
        elif t == 0 and p == 0:
            buckets["tn"].append(i)
        elif t == 0 and p == 1:
            buckets["fp"].append(i)
        else:
            buckets["fn"].append(i)
    return buckets


def error_counts(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    """Compact error summary: confusion + macro-F1 + per-class report."""
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    return {
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "macro_f1": round(macro_f1(y_true, y_pred), 6),
        "per_class": per_class_report(y_true, y_pred),
    }


def representative_examples(
    pairs: pd.DataFrame,
    indices: list[int],
    n: int,
    *,
    seed: int = 42,
    fields: tuple[str, ...] = (
        "resume_role",
        "resume_seniority",
        "resume_years_experience",
        "resume_industry",
        "job_title",
        "job_seniority",
        "job_industry",
    ),
) -> list[dict[str, Any]]:
    """Sample up to ``n`` deterministic examples summarised by derived fields.

    Explicitly does NOT include summary text, bullets, or descriptions.
    """
    if not indices:
        return []
    rng = random.Random(seed)
    sample = rng.sample(indices, min(n, len(indices)))
    out: list[dict[str, Any]] = []
    for idx in sample:
        row = pairs.iloc[idx]
        out.append({
            field: (str(row.get(field)) if pd.notna(row.get(field)) else None)
            for field in fields
        })
    return out


def pattern_counts(
    pairs: pd.DataFrame,
    indices: list[int],
    *,
    label: int,
    cosine_field: str = "minilm_cosine",
) -> dict[str, Any]:
    """Count compact derived patterns within an error bucket.

    ``patterns`` is a list of (name, predicate) where the predicate receives
    the row and decides membership.  Only the subset of rows returned by
    ``representative_examples``-friendly predicates is counted; every rule is
    a *derived signal heuristic*, never a claim about causes.
    """
    patterns: list[tuple[str, Callable[[pd.Series], bool]]] = [
        ("high_cosine_but_wrong",
         lambda r: float(r.get(cosine_field, 0.0) or 0.0) >= 0.5),
        ("overlap_required_high_but_wrong",
         lambda r: float(r.get("keyword_overlap_required", 0.0) or 0.0) >= 0.6),
        ("role_mismatch",
         lambda r: str(r.get("resume_role", "")).lower()
         != str(r.get("job_title", "")).lower()),
        ("industry_mismatch",
         lambda r: str(r.get("resume_industry", "")).lower()
         != str(r.get("job_industry", "")).lower()),
        ("seniority_mismatch",
         lambda r: str(r.get("resume_seniority", "")).lower()
         != str(r.get("job_seniority", "")).lower()),
    ]

    counts: dict[str, int] = {}
    within_bucket = 0
    for idx in indices:
        row = pairs.iloc[idx]
        within_bucket += 1
        for name, predicate in patterns:
            if predicate(row):
                counts[name] = counts.get(name, 0) + 1

    return {
        "bucket": "positive" if label == 1 else "negative",
        "examples_in_bucket": within_bucket,
        "pattern_counts": counts,
        "note": (
            "Hypotheses are derived-signal heuristics for report structuring, "
            "not causal conclusions."
        ),
    }
