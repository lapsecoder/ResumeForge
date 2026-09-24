"""Evaluation metrics for the candidate-matching experiments.

Only metric *definitions* live here. Primary metric is macro-F1 (both classes
count equally despite the explicit positive sampling); accuracy, balanced
accuracy, per-class P/R/F1 and behaviour of probabilistic models
(ROC-AUC / PR-AUC) are secondary or diagnostic.

Functions are pure: no state, no training, no sklearn. They are deliberately
kept in the standard library so they can be unit-tested offline and shared by
any 6E harness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

PRIMARY_METRIC = "macro_f1"


@dataclass(frozen=True)
class MetricSpec:
    """One metric in the committed evaluation set."""

    name: str
    kind: str  # "classification" | "ranking" | "diagnostic"
    requires_probability: bool
    scope: str  # "primary" | "secondary" | "diagnostic"
    description: str


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "macro_f1", "classification", False, "primary",
        "Mean of positive-class and negative-class F1; used to pick the best model.",
    ),
    MetricSpec(
        "accuracy", "classification", False, "secondary",
        "Overall fraction correct. Reported, never used to compare models.",
    ),
    MetricSpec(
        "balanced_accuracy", "classification", False, "secondary",
        "Mean of positive and negative recall; robust to class proportions.",
    ),
    MetricSpec(
        "per_class_f1", "classification", False, "secondary",
        "F1 reported separately for the positive and negative class.",
    ),
    MetricSpec(
        "per_class_precision", "classification", False, "secondary",
        "Precision reported separately for each class.",
    ),
    MetricSpec(
        "per_class_recall", "classification", False, "secondary",
        "Recall reported separately for each class.",
    ),
    MetricSpec(
        "confusion_matrix", "classification", False, "secondary",
        "TP/FP/FN/TN for the binary threshold applied to predictions.",
    ),
    MetricSpec(
        "roc_auc", "ranking", True, "diagnostic",
        "Ranking quality over the negative/positive split for probabilistic models.",
    ),
    MetricSpec(
        "pr_auc", "ranking", True, "diagnostic",
        "Precision-recall AUC; informative under the sampled-negative framing.",
    ),
)


def confusion_counts(
    y_true: Sequence[Any], y_pred: Sequence[Any]
) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, fn, tn)`` treating 1 as the positive label."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    pairs = list(zip([int(v) for v in y_true], [int(v) for v in y_pred], strict=True))
    if len(pairs) == 0:
        raise ValueError("cannot compute metrics on an empty prediction set")
    tp = sum(1 for t, p in pairs if t == 1 and p == 1)
    fp = sum(1 for t, p in pairs if t == 0 and p == 1)
    fn = sum(1 for t, p in pairs if t == 1 and p == 0)
    tn = sum(1 for t, p in pairs if t == 0 and p == 0)
    return tp, fp, fn, tn


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    return _safe_ratio(tp + tn, tp + fp + fn + tn)


def recall_positive(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    tp, fp, fn, _ = confusion_counts(y_true, y_pred)
    return _safe_ratio(tp, tp + fn)


def recall_negative(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    return _safe_ratio(tn, tn + fp)


def precision_positive(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    tp, fp, _, _ = confusion_counts(y_true, y_pred)
    return _safe_ratio(tp, tp + fp)


def precision_negative(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    return _safe_ratio(tn, tn + fn)


def f1_positive(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    precision = precision_positive(y_true, y_pred)
    recall = recall_positive(y_true, y_pred)
    return _safe_ratio(2 * precision * recall, precision + recall)


def f1_negative(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    precision = precision_negative(y_true, y_pred)
    recall = recall_negative(y_true, y_pred)
    return _safe_ratio(2 * precision * recall, precision + recall)


def macro_f1(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Primary metric: unweighted mean of positive and negative F1."""
    return (f1_positive(y_true, y_pred) + f1_negative(y_true, y_pred)) / 2.0


def balanced_accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Mean of positive and negative recall."""
    return (recall_positive(y_true, y_pred) + recall_negative(y_true, y_pred)) / 2.0


def per_class_report(
    y_true: Sequence[Any], y_pred: Sequence[Any]
) -> dict[str, dict[str, float]]:
    """``{class: {precision, recall, f1}}`` keyed by ``"positive"``/``"negative"``."""
    return {
        "positive": {
            "precision": precision_positive(y_true, y_pred),
            "recall": recall_positive(y_true, y_pred),
            "f1": f1_positive(y_true, y_pred),
        },
        "negative": {
            "precision": precision_negative(y_true, y_pred),
            "recall": recall_negative(y_true, y_pred),
            "f1": f1_negative(y_true, y_pred),
        },
    }
