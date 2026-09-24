"""Deterministic Phase-5A baseline evaluation.

Builds the SAME pairs and job_grouped split as the 6E experiments,
computes match_resume_to_job scores, derives a binary threshold from the
training split only, and evaluates on the frozen test set.

Run:
    cd backend && python -m ml.deterministic_baseline.evaluate
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import pandas as pd

from ml.config import ARTIFACTS_DIR
from ml.data.pairs import build_pairs
from ml.deterministic_baseline.adapter import (
    compute_phase5a_scores,
    derive_threshold_on_train,
    score_to_label,
)
from ml.loader import load_bundle
from ml.metrics import (
    accuracy,
    balanced_accuracy,
    confusion_counts,
    macro_f1,
    per_class_report,
)
from ml.splits import job_group_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("deterministic_baseline")

ARTIFACT_DIR = ARTIFACTS_DIR / "phase_6e"
SEED = 42
NEGATIVES_PER_JOB = 30
REFERENCE_DATE = date(2026, 1, 1)


def run() -> dict[str, Any]:
    _ARTIFACT_DIR = ARTIFACT_DIR
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle()
    resumes_df: pd.DataFrame = bundle.resumes
    jobs_df: pd.DataFrame = bundle.jobs
    matches_df: pd.DataFrame = bundle.matches

    logger.info(
        "Loaded dataset: %d resumes, %d jobs, %d matches",
        len(resumes_df), len(jobs_df), len(matches_df),
    )

    full_pairs = build_pairs(
        resumes_df, matches_df,
        negatives_per_job=NEGATIVES_PER_JOB, seed=SEED,
    )
    logger.info("Built %d pairs (pos=%d, neg=%d)",
                len(full_pairs),
                int(full_pairs["relevant"].sum()),
                int((1 - full_pairs["relevant"]).sum()))

    job_assign = job_group_split(jobs_df, seed=SEED, sizes=(0.7, 0.1, 0.2))
    full_pairs = full_pairs.copy()
    full_pairs["_split"] = full_pairs["job_id"].map(job_assign)

    train_pairs = full_pairs[full_pairs["_split"] == "train"].copy()
    val_pairs = full_pairs[full_pairs["_split"] == "val"].copy()
    test_pairs = full_pairs[full_pairs["_split"] == "test"].copy()

    logger.info("Split sizes: train=%d, val=%d, test=%d",
                len(train_pairs), len(val_pairs), len(test_pairs))
    assert len(train_pairs) == 106920, f"Expected 106920 train, got {len(train_pairs)}"
    assert len(val_pairs) == 15000, f"Expected 15000 val, got {len(val_pairs)}"
    assert len(test_pairs) == 28080, f"Expected 28080 test, got {len(test_pairs)}"

    resumes_lookup = resumes_df.set_index("resume_id").to_dict("index")
    jobs_lookup = jobs_df.set_index("job_id").to_dict("index")

    logger.info("Computing Phase-5A scores on training split...")
    train_scores = compute_phase5a_scores(train_pairs, resumes_lookup, jobs_lookup)
    y_train = train_pairs["relevant"].astype(int).tolist()

    logger.info("Deriving threshold on training split (macro-F1 maximisation)...")
    frozen_threshold, threshold_curve = derive_threshold_on_train(train_scores, y_train)
    logger.info("Frozen threshold: %.1f", frozen_threshold)

    logger.info("Computing Phase-5A scores on validation split...")
    val_scores = compute_phase5a_scores(val_pairs, resumes_lookup, jobs_lookup)
    y_val = val_pairs["relevant"].astype(int).tolist()
    val_preds = score_to_label(val_scores, frozen_threshold)

    logger.info("Computing Phase-5A scores on test split...")
    test_scores = compute_phase5a_scores(test_pairs, resumes_lookup, jobs_lookup)
    y_test = test_pairs["relevant"].astype(int).tolist()
    test_preds = score_to_label(test_scores, frozen_threshold)

    def _eval(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
        tp, fp, fn, tn = confusion_counts(y_true, y_pred)
        pc = per_class_report(y_true, y_pred)
        return {
            "macro_f1": round(macro_f1(y_true, y_pred), 4),
            "accuracy": round(accuracy(y_true, y_pred), 4),
            "balanced_accuracy": round(balanced_accuracy(y_true, y_pred), 4),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "per_class": pc,
        }

    train_preds = score_to_label(train_scores, frozen_threshold)
    train_eval = _eval(y_train, train_preds)
    val_eval = _eval(y_val, val_preds)
    test_eval = _eval(y_test, test_preds)

    def _none_fraction(scores: list[float | None]) -> float:
        if not scores:
            return 0.0
        return sum(1 for s in scores if s is None) / len(scores)

    none_frac_train = _none_fraction(train_scores)
    none_frac_test = _none_fraction(test_scores)

    result = {
        "split_type": "job_grouped",
        "frozen_threshold": frozen_threshold,
        "train_samples": len(train_pairs),
        "val_samples": len(val_pairs),
        "test_samples": len(test_pairs),
        "score_range": {
            "min": round(min(s for s in test_scores if s is not None), 4),
            "max": round(max(s for s in test_scores if s is not None), 4),
        },
        "none_score_fraction": {
            "train": round(none_frac_train, 4),
            "test": round(none_frac_test, 4),
        },
        "train_eval": train_eval,
        "val_eval": val_eval,
        "test_eval": test_eval,
        "threshold_curve_top10": dict(
            sorted(threshold_curve.items(), key=lambda kv: kv[1], reverse=True)[:10]
        ),
        "note": (
            "Threshold derived from job_grouped TRAIN split only (maximise "
            "macro-F1). Evaluated once on frozen job_grouped TEST. "
            "Phase-5A overall_score reduces to skill-coverage weighting "
            "(required_skill 50 + preferred_skill 15 of 65 total) because "
            "the synthetic dataset has no structured experience dates, "
            "education requirements, or certification requirements."
        ),
    }

    out_path = _ARTIFACT_DIR / "deterministic_baseline_metrics.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved deterministic baseline metrics to %s", out_path)

    cm_path = _ARTIFACT_DIR / "deterministic_baseline_confusion_matrix.json"
    with open(cm_path, "w") as f:
        json.dump(test_eval["confusion"], f, indent=2)
    logger.info("Saved confusion matrix to %s", cm_path)

    logger.info("=" * 80)
    logger.info("DETERMINISTIC PHASE-5A BASELINE RESULTS (job_grouped)")
    logger.info("=" * 80)
    logger.info("Frozen threshold: %.1f", frozen_threshold)
    logger.info(
        "TRAIN : macro-F1=%.4f, accuracy=%.4f",
        train_eval["macro_f1"], train_eval["accuracy"],
    )
    logger.info(
        "VAL   : macro-F1=%.4f, accuracy=%.4f",
        val_eval["macro_f1"], val_eval["accuracy"],
    )
    logger.info(
        "TEST  : macro-F1=%.4f, accuracy=%.4f",
        test_eval["macro_f1"], test_eval["accuracy"],
    )
    logger.info("Test confusion: %s", test_eval["confusion"])
    logger.info("=" * 80)

    return result


if __name__ == "__main__":
    run()
