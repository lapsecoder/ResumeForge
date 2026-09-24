"""Feature-importance extraction for the 6E trained models.

The 6E estimators are all linear: raw LogisticRegression/LinearSVC on
TF-IDF, or ``imputer -> scaler -> LogisticRegression`` pipelines on the
structured/combined features.  Coefficient extraction is therefore exact,
deterministic, and reproducible from the saved artifacts (no refitting).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class CoefficientEstimator(Protocol):
    """Minimal surface of the sklearn estimators used in 6E."""

    def predict(self, X: Any) -> Any: ...


def _clf_from_pipeline(pipeline: Pipeline) -> Any:
    """Return the fitted classifier step of a 6E feature pipeline."""
    if not isinstance(pipeline, Pipeline) or "clf" not in pipeline.named_steps:
        raise TypeError("expected a Pipeline with a 'clf' step")
    return pipeline["clf"]


def _scaler_std(estimator: Any) -> np.ndarray | None:
    """Return the per-feature standard deviation used by a scaler step, if any."""
    if not isinstance(estimator, Pipeline) or "scaler" not in estimator.named_steps:
        return None
    scaler = estimator["scaler"]
    if not hasattr(scaler, "scale_"):
        return None
    scale_arr: np.ndarray = np.asarray(scaler.scale_).ravel().copy()
    scale_arr[scale_arr == 0.0] = 1.0  # constant features: no scaling effect
    return scale_arr


def _coefficients(estimator: Any, feature_names: list[str]) -> pd.Series:
    """Pull the linear coefficients aligned to ``feature_names``."""
    if isinstance(estimator, Pipeline):
        clf = _clf_from_pipeline(estimator)
    else:
        clf = estimator

    coef = getattr(clf, "coef_", None)
    if coef is None or coef.size == 0:
        raise ValueError("estimator has no coef_; not a linear model")
    vector = np.asarray(coef).ravel()
    if vector.shape[0] != len(feature_names):
        raise ValueError(
            f"coefficient count {vector.shape[0]} != feature count {len(feature_names)}"
        )
    return pd.Series(vector, index=feature_names, dtype=float)


def extract_importances(
    estimator: Any,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    """Return ranked feature importances for a 6E linear estimator.

    ``coefficient`` is the raw model coefficient (for standardized pipelines
    this is the log-odds effect of a one-standard-deviation change, so
    features are directly comparable).  ``coef_per_original_unit`` rescales to
    the effect of one unit of the original feature.  Ranks by
    ``abs_coefficient`` == |coefficient|.
    """
    coef = _coefficients(estimator, feature_names)
    diagonal = _scaler_std(estimator)

    rows: list[dict[str, Any]] = []
    for name in feature_names:
        value = float(coef[name])
        scale = 1.0
        if diagonal is not None:
            scale = float(diagonal[list(coef.index).index(name)])
        rows.append({
            "feature": name,
            "coefficient": round(value, 6),
            "coef_per_original_unit": round(value / scale, 6),
            "std": round(scale, 6) if diagonal is not None else None,
            "abs_coefficient": round(abs(value), 6),
        })

    return sorted(rows, key=lambda row: row["abs_coefficient"], reverse=True)


def importance_by_family(importances: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate importances by feature family for the report."""
    from ml.phase_6f.specs import FEATURE_FAMILY

    family_abs: dict[str, float] = {}
    family_features: dict[str, int] = {}
    for row in importances:
        family = FEATURE_FAMILY.get(row["feature"], "unknown")
        family_abs[family] = family_abs.get(family, 0.0) + row["abs_coefficient"]
        family_features[family] = family_features.get(family, 0) + 1

    return {
        "by_family": {
            family: {
                "sum_abs_coefficient": round(value, 6),
                "n_features": family_features[family],
                "mean_abs_coefficient": (
                    round(value / family_features[family], 6)
                    if family_features[family]
                    else 0.0
                ),
            }
            for family, value in sorted(family_abs.items())
        }
    }
