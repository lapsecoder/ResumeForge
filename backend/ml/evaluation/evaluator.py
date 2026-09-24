"""Evaluation implementation for Phase 6E.

Evaluates trained models against held-out test data and produces
EvaluationReport instances using the committed metric set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ml.evaluation.contracts import (
    EvaluationConfig,
    EvaluationReport,
    default_evaluation_config,
)
from ml.metrics import (
    accuracy,
    balanced_accuracy,
    confusion_counts,
    macro_f1,
    per_class_report,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationContext:
    """Context for a single evaluation run."""

    tier: int
    tier_name: str
    split_name: str
    config: EvaluationConfig = field(default_factory=default_evaluation_config)


class ModelEvaluator:
    """Evaluates a fitted model on a test feature matrix."""

    def evaluate(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        *,
        config: EvaluationConfig | None = None,
        tier: int = 0,
        tier_name: str = "",
        split_name: str = "",
    ) -> EvaluationReport:
        if config is None:
            config = default_evaluation_config()

        if len(y_test) == 0:
            raise ValueError("cannot evaluate on an empty test set")
        if model is None:
            raise ValueError("cannot evaluate with a None model")

        X_arr = (
            X_test.values.astype(float)
            if len(X_test.columns) > 0
            else np.zeros((len(y_test), 0))
        )

        if hasattr(model, "predict"):
            if hasattr(model, "estimator"):
                raw_preds = model.estimator.predict(X_arr)
            else:
                raw_preds = model.predict(X_arr)
            y_pred = [int(p) for p in raw_preds]
        else:
            y_pred = [int(p) for p in model.predict(X_arr)]

        y_true = [int(v) for v in y_test]

        tp, fp, fn, tn = confusion_counts(y_true, y_pred)
        acc = accuracy(y_true, y_pred)
        bal_acc = balanced_accuracy(y_true, y_pred)
        mf1 = macro_f1(y_true, y_pred)
        pc = per_class_report(y_true, y_pred)

        metrics: dict[str, float] = {
            "macro_f1": mf1,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
        }

        for class_name, vals in pc.items():
            for metric_name, val in vals.items():
                metrics[f"{class_name}_{metric_name}"] = val

        report = EvaluationReport(
            metrics=metrics,
            confusion={"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            per_class=pc,
            metadata={
                "tier": tier,
                "tier_name": tier_name,
                "split_name": split_name,
                "n_test": len(y_true),
                "n_features": len(X_test.columns),
            },
        )

        logger.info(
            "Tier %d (%s) on %s: macro_f1=%.4f, accuracy=%.4f, n=%d",
            tier, tier_name, split_name, mf1, acc, len(y_true),
        )

        return report


def default_evaluator() -> ModelEvaluator:
    return ModelEvaluator()
