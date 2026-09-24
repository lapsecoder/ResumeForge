"""Experiment configuration and versioning.

One :class:`ExperimentConfig` fully determines a reproducible run: dataset
snapshot, seed, negative sampling, split strategy/sizes and every pipeline
version. Two configs with the same :meth:`ExperimentConfig.fingerprint` are
the same experiment; bump the relevant version field when a definition
changes so old fingerprints stop matching.

The experiment tiers (0-5) are *declared* here. Phase 6D does not run them;
6E implements the harness behind these signatures.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from ml.config import DATASET_ID, DATASET_REVISION
from ml.metrics import PRIMARY_METRIC

VERSION_PREPROCESSING = "1.0.0"
VERSION_FEATURES = "1.0.0"
VERSION_SPLITS = "1.0.0"
VERSION_EXPERIMENTS = "1.0.0"

SplitStrategy = Literal["job_grouped", "strict_both", "resume_grouped"]


@dataclass(frozen=True)
class TierSpec:
    """One experiment tier: feature families, model and its role."""

    tier: int
    name: str
    model: str
    feature_families: tuple[str, ...]
    purpose: str


TIERS: tuple[TierSpec, ...] = (
    TierSpec(
        0, "majority baseline", "majority", (),
        "Predicts the majority class everywhere; establishes the chance floor.",
    ),
    TierSpec(
        1, "TF-IDF + logistic regression",
        "logistic_regression", ("resume_text", "job_text"),
        "Cheap text-only model; measure of vocabulary overlap alone.",
    ),
    TierSpec(
        2, "TF-IDF + linear SVM",
        "linear_svm", ("resume_text", "job_text"),
        "Text-only with a margin; contrast to the probabilistic baseline.",
    ),
    TierSpec(
        3, "MiniLM embeddings + classical head",
        "classical", ("minilm",),
        "Local semantic embeddings, no fine-tuning. Semantic-only signal.",
    ),
    TierSpec(
        4, "structured features + classical head",
        "classical",
        ("structured_resume", "structured_job", "deterministic_match",
         "ats_6a", "ats_6b"),
        "ResumeForge's deterministic parser/ATS features without text tokens.",
    ),
    TierSpec(
        5, "combined (text + semantic + structured)",
        "classical", tuple(
            ("resume_text", "job_text", "minilm", "structured_resume",
             "structured_job", "deterministic_match", "ats_6a", "ats_6b")
        ),
        "All approved families; the reference configuration for 6E.",
    ),
)


class ExperimentConfig(BaseModel):
    """Fully-specified reproducibility record for one experiment run."""

    dataset_id: str = DATASET_ID
    dataset_revision: str = DATASET_REVISION
    seed: int = 42
    negatives_per_job: int = 30
    split_strategy: SplitStrategy = "job_grouped"
    split_sizes: tuple[float, float, float] = (0.7, 0.1, 0.2)
    preprocessing_version: str = VERSION_PREPROCESSING
    feature_version: str = VERSION_FEATURES
    split_version: str = VERSION_SPLITS
    experiment_version: str = VERSION_EXPERIMENTS
    primary_metric: str = PRIMARY_METRIC
    tiers: tuple[int, ...] = Field(
        default_factory=lambda: tuple(tier.tier for tier in TIERS)
    )

    @classmethod
    def default(cls) -> "ExperimentConfig":
        return cls()

    def fingerprint(self) -> str:
        """Short content hash: identical configs must hash identically."""
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def validate_tiers(self) -> None:
        known = {tier.tier for tier in TIERS}
        unknown = [tier for tier in self.tiers if tier not in known]
        if unknown:
            raise ValueError(f"unknown tiers in config: {unknown}")


# Modest, documented tuning grids. Declared only: 6E runs them as
# leakage-safe GroupKFold(folds grouped by job) loops inside the train split -
# the test split is never touched during tuning.
MODEST_GRID: dict[str, dict[str, tuple[Any, ...]]] = {
    "logistic_regression": {
        "C": (0.1, 1.0, 10.0),
        "class_weight": ("balanced",),
    },
    "linear_svm": {
        "C": (0.1, 1.0, 10.0),
    },
    "classical": {
        # feature-side: max_features on TF-IDF, min_df on the vocabulary
        "tfidf_max_features": (500, 2000, None),
        "tfidf_min_df": (1, 3),
        "scale_features": (True, False),
    },
}
