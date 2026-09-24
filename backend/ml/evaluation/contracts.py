"""Evaluation contracts for Phase 6E.

Defines the shape every evaluation must return so metrics stay comparable
across tiers. EvaluationReport is a pure data class; the actual evaluator
that produces reports lives in :mod:`ml.evaluation.evaluator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ml.metrics import PRIMARY_METRIC as PRIMARY_METRIC

DEFAULT_METRICS = (
    "macro_f1",
    "accuracy",
    "balanced_accuracy",
    "per_class_f1",
    "per_class_precision",
    "per_class_recall",
    "confusion_matrix",
)


@dataclass(frozen=True)
class EvaluationConfig:
    """Which metrics an evaluation must produce."""

    metrics: tuple[str, ...] = field(default_factory=lambda: DEFAULT_METRICS)
    primary_metric: str = PRIMARY_METRIC


@dataclass(frozen=True)
class EvaluationReport:
    """Committed result shape of one evaluation run."""

    metrics: dict[str, float]
    confusion: dict[str, int]
    per_class: dict[str, dict[str, float]]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.primary_missing:
            raise ValueError("report missing the primary metric")
        if self.confusion_keys_invalid:
            raise ValueError("confusion must contain tp/fp/fn/tn")

    @property
    def primary_missing(self) -> bool:
        return PRIMARY_METRIC not in self.metrics

    @property
    def confusion_keys_invalid(self) -> bool:
        return not all(key in self.confusion for key in ("tp", "fp", "fn", "tn"))


class Evaluator(Protocol):
    """Evaluates a trained model on a pair-table and returns a report."""

    def evaluate(
        self,
        model: Any,
        X: Any,
        y: Any,
        *,
        config: EvaluationConfig,
    ) -> EvaluationReport: ...


def default_evaluation_config() -> EvaluationConfig:
    return EvaluationConfig()
