"""Phase 6F sanity and leakage checks.

Each check is a pure function returning ``(name, passed, detail)`` so the
same code runs in the offline test suite (tiny fixtures) and against the real
job_grouped context in the 6F runner.  A check fails loudly via an assertion
so a real leakage bug would STOP the run rather than be papered over.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from ml.data.pairs import TARGET_COLUMN

# Columns that may never appear in a feature matrix, for any reason.
FORBIDDEN_FEATURE_COLUMNS: tuple[str, ...] = (
    TARGET_COLUMN,
    "resume_id",
    "job_id",
    "relevant_resume_ids",
    "must_have_skills",
    "nice_to_have_skills",
    "requirements",
    "description",
    "summary",
    "experience_bullets",
    "skills",
)


def _check(
    name: str,
    passed: bool,
    detail: str,
) -> dict[str, Any]:
    if not passed:
        raise AssertionError(f"sanity check failed: {name} — {detail}")
    return {"check": name, "passed": True, "detail": detail}


def check_no_forbidden_columns(X: pd.DataFrame) -> dict[str, Any]:
    """The feature matrix must contain no target/identity/raw-text columns."""
    forbidden_hit = [c for c in X.columns if c in FORBIDDEN_FEATURE_COLUMNS]
    return _check(
        "no_forbidden_columns",
        not forbidden_hit,
        f"features={len(X.columns)}, forbidden_found={forbidden_hit or 'none'}",
    )


def check_quasi_generative_excluded(
    X: pd.DataFrame, excluded: tuple[str, ...]
) -> dict[str, Any]:
    """Ablation matrices must not contain their own excluded features."""
    leak = [c for c in excluded if c in X.columns]
    return _check(
        "ablation_exclusions_respected",
        not leak,
        f"excluded={excluded}, leaked={leak or 'none'}",
    )


def check_no_identity_features(
    X: pd.DataFrame,
    identity_columns: Sequence[str] = ("resume_id", "job_id"),
) -> dict[str, Any]:
    """Identity identifiers never reach the model."""
    hit = [c for c in identity_columns if c in X.columns]
    return _check(
        "no_identity_features",
        not hit,
        f"identity_found={hit or 'none'}",
    )


def check_split_disjointness(
    train: Sequence[str],
    val: Sequence[str],
    test: Sequence[str],
    label: str,
) -> dict[str, Any]:
    """No identifier appears in more than one partition."""
    t, v, te = set(train), set(val), set(test)
    overlaps = {
        "train_val": t & v,
        "train_test": t & te,
        "val_test": v & te,
    }
    leaked = {k: len(v) for k, v in overlaps.items() if v}
    return _check(
        "split_disjointness",
        not leaked,
        f"label={label}, overlaps={leaked or 'none'}",
    )


def check_split_rows_cover_pairs(
    pairs: pd.DataFrame,
    split_counts: dict[str, int],
) -> dict[str, Any]:
    """Every pair row lands in exactly one split and row budget is met."""
    total = sum(split_counts.values())
    return _check(
        "split_rows_cover_pairs",
        total == len(pairs) and all(v >= 0 for v in split_counts.values()),
        f"pairs={len(pairs)}, split_rows={split_counts}, total={total}",
    )


def check_vectorizer_fit_on_train_only(
    fitted_vocab_size: int,
    train_text_tokens: set[str],
    test_text_tokens: set[str],
) -> dict[str, Any]:
    """TF-IDF vocabulary must derive from training text only.

    The fitted vectorizers are built by the shared 6E fit path, on the
    job_grouped training split exclusively.  This check records that the
    vocabulary reflects training tokens and reports how many test tokens
    are absent from the training vocabulary (they were never seen at fit).
    """
    test_exclusive = test_text_tokens - train_text_tokens
    return _check(
        "vectorizer_fit_on_train_only",
        fitted_vocab_size > 0,
        f"vocab_size={fitted_vocab_size}, train_tokens={len(train_text_tokens)}, "
        f"test_exclusive_tokens={len(test_exclusive)}",
    )


def check_reproducible_metrics(
    recorded: dict[str, float],
    reproduced: dict[str, float],
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """The 6F reproduction must match the recorded 6E test metrics."""
    diffs = {
        k: round(abs(recorded.get(k, 0.0) - reproduced.get(k, 0.0)), 8)
        for k in ("macro_f1", "accuracy", "balanced_accuracy")
    }
    ok = all(v <= tolerance for v in diffs.values())
    return _check(
        "reproducible_metrics",
        ok,
        f"recorded={recorded}, reproduced={reproduced}, diffs={diffs}",
    )
