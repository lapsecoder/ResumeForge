"""Training implementation for Phase 6E.

Replaces the Phase 6D stub with actual fitting of sklearn-compatible
classifiers.  Each tier instantiates its own pipeline from the experiment
plan's model specification and feature families.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ml.experiments.config import TIERS, ExperimentConfig
from ml.splits import SplitManifest

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    """A fitted binary classifier over pair feature tables."""

    estimator: Any
    feature_names: list[str]
    tier: int
    model_type: str
    train_samples: int
    train_time_s: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, X: Any) -> list[int]:
        preds = self.estimator.predict(X)
        return [int(p) for p in preds]

    def predict_proba(self, X: Any) -> list[tuple[float, float]]:
        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(X)
            return [(float(p[0]), float(p[1])) for p in proba]
        if hasattr(self.estimator, "decision_function"):
            scores = self.estimator.decision_function(X)
            exp_neg = np.exp(-scores)
            prob1 = 1.0 / (1.0 + exp_neg)
            return [(float(1.0 - p), float(p)) for p in prob1]
        return [(0.5, 0.5)] * len(X)


@dataclass(frozen=True)
class ExperimentPlan:
    """Everything a single run needs: config + tier + data partition."""

    config: ExperimentConfig
    tier: int
    model_spec: str
    manifest: SplitManifest

    def describe(self) -> dict[str, Any]:
        tier_spec = next(t for t in TIERS if t.tier == self.tier)
        return {
            "tier": self.tier,
            "name": tier_spec.name,
            "model": self.model_spec,
            "fingerprint": self.config.fingerprint(),
            "split_strategy": self.config.split_strategy,
            "split_summary": self.manifest.summary(),
        }


def _build_majority_classifier() -> Any:

    class _MajorityClassifier:
        def __init__(self) -> None:
            self.majority_class_ = 0
            self.classes_ = np.array([0, 1])

        def fit(self, X: Any, y: Any) -> "_MajorityClassifier":
            vals, counts = np.unique(y, return_counts=True)
            self.majority_class_ = int(vals[np.argmax(counts)])
            self.classes_ = vals
            return self

        def predict(self, X: Any) -> np.ndarray:
            n = len(X)
            return np.full(n, self.majority_class_)

        def predict_proba(self, X: Any) -> np.ndarray:
            n = len(X)
            probs: Any
            if self.majority_class_ == 0:
                probs = np.column_stack((np.ones(n), np.zeros(n)))
            else:
                probs = np.column_stack((np.zeros(n), np.ones(n)))
            return cast(np.ndarray, probs)

        def decision_function(self, X: Any) -> np.ndarray:
            n = len(X)
            return np.full(n, float(-1 if self.majority_class_ == 0 else 1))

    return _MajorityClassifier()


def _build_logistic_regression(
    C: float = 1.0, class_weight: str = "balanced", max_iter: int = 1000,
) -> Any:
    return LogisticRegression(
        C=C, class_weight=class_weight, max_iter=max_iter,
        solver="lbfgs", random_state=42,
    )


def _build_linear_svm(C: float = 1.0, max_iter: int = 2000) -> Any:
    return LinearSVC(
        C=C, class_weight="balanced", max_iter=max_iter,
        random_state=42, dual="auto",
    )


def train_tfidf_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_type: str = "logistic_regression",
    C: float = 1.0,
) -> tuple[Any, list[str]]:
    """Train a classifier on TF-IDF features."""
    cols = list(X_train.columns)
    X_arr = X_train.values.astype(float)

    if model_type == "logistic_regression":
        clf = _build_logistic_regression(C=C)
    elif model_type == "linear_svm":
        clf = _build_linear_svm(C=C)
    else:
        raise ValueError(f"unknown model_type: {model_type}")

    clf.fit(X_arr, y_train)
    return clf, cols


def _logistic_pipeline(C: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                C=C, class_weight="balanced", max_iter=1000, random_state=42
            ),
        ),
    ])


def train_semantic_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    C: float = 1.0,
) -> tuple[Any, list[str]]:
    """Train a classifier on semantic embedding features."""
    cols = list(X_train.columns)
    X_arr = X_train.values.astype(float)

    pipeline = _logistic_pipeline(C)
    pipeline.fit(X_arr, y_train)
    return pipeline, cols


def train_structured_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    C: float = 1.0,
) -> tuple[Any, list[str]]:
    """Train a classifier on structured features."""
    cols = list(X_train.columns)
    X_arr = X_train.values.astype(float)

    pipeline = _logistic_pipeline(C)
    pipeline.fit(X_arr, y_train)
    return pipeline, cols


def train_combined_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    C: float = 1.0,
) -> tuple[Any, list[str]]:
    """Train a classifier on combined features."""
    cols = list(X_train.columns)
    X_arr = X_train.values.astype(float)

    pipeline = _logistic_pipeline(C)
    pipeline.fit(X_arr, y_train)
    return pipeline, cols


def train(
    plan: ExperimentPlan, X_train: pd.DataFrame, y_train: np.ndarray
) -> TrainedModel:
    """Fit the model for ``plan`` on the provided feature matrix and labels."""
    start = time.time()
    tier_spec = next(t for t in TIERS if t.tier == plan.tier)

    if plan.tier == 0:
        clf = _build_majority_classifier()
        X_fit = (
            X_train.values.astype(float)
            if len(X_train.columns) > 0
            else np.zeros((len(y_train), 0))
        )
        clf.fit(X_fit, y_train)
        model_type = "majority"
        feature_names = list(X_train.columns)
    elif plan.tier in (1, 2):
        model_type = tier_spec.model
        clf, feature_names = train_tfidf_classifier(
            X_train, y_train, model_type=model_type, C=1.0
        )
    elif plan.tier == 3:
        clf, feature_names = train_semantic_classifier(X_train, y_train, C=1.0)
        model_type = "logistic_regression"
    elif plan.tier == 4:
        clf, feature_names = train_structured_classifier(X_train, y_train, C=1.0)
        model_type = "logistic_regression"
    elif plan.tier == 5:
        clf, feature_names = train_combined_classifier(X_train, y_train, C=1.0)
        model_type = "logistic_regression"
    else:
        raise ValueError(f"unknown tier: {plan.tier}")

    elapsed = time.time() - start
    logger.info(
        "Tier %d (%s) trained: %d samples, %d features, %.2fs",
        plan.tier, tier_spec.name, len(y_train), len(feature_names), elapsed,
    )

    return TrainedModel(
        estimator=clf,
        feature_names=feature_names,
        tier=plan.tier,
        model_type=model_type,
        train_samples=len(y_train),
        train_time_s=elapsed,
        metadata={
            "tier_name": tier_spec.name,
            "feature_families": list(tier_spec.feature_families),
        },
    )
