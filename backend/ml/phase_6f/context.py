"""Phase 6F context: dataset access plus per-tier feature rebuilds.

Everything here reuses the Phase 6E pipeline untouched (loaders, splits,
feature builders, trainer).  The goal of 6F is analysis of what was trained,
so all delivery splits use the same ``job_grouped`` split and the same
feature builders as the 6E run — the primary benchmark for the analysis.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.config import ARTIFACTS_DIR
from ml.data.pairs import build_pairs
from ml.evaluation.evaluator import default_evaluator
from ml.experiments.config import ExperimentConfig
from ml.features.semantic import SemanticEmbedder, build_semantic_features
from ml.features.text import FittedVectorizer, JobTextVectorizer, ResumeTextVectorizer
from ml.loader import load_bundle
from ml.models.trainer import ExperimentPlan, train
from ml.scripts.run_phase_6e import (
    _build_features_for_tier,
    _join_pair_data,
    _split_pairs,
)
from ml.splits import SplitManifest, job_group_split

SEED = 42
NEGATIVES_PER_JOB = 30
PHASE_6E_DIR = ARTIFACTS_DIR / "phase_6e"
PHASE_6F_DIR = ARTIFACTS_DIR / "phase_6f"


class JobGroupedContext:
    """Wraps the job_grouped split of the enriched pair table.

    Owns the pairs (train/val/test), the fitted TF-IDF vectorizers, and the
    reconstructed feature matrixes so analysis modules share one consistent
    view of the data.
    """

    def __init__(
        self,
        pairs: dict[str, pd.DataFrame],
        fitted_resume_vec: FittedVectorizer,
        fitted_job_vec: FittedVectorizer,
        semantic_features: dict[str, pd.DataFrame],
    ) -> None:
        self.pairs = pairs
        self.fitted_resume_vec = fitted_resume_vec
        self.fitted_job_vec = fitted_job_vec
        self.semantic_features = semantic_features

    def y_train(self) -> list[int]:
        raw = self.pairs["train"]["relevant"].astype(int).tolist()
        return [int(v) for v in raw]

    def y_test(self) -> list[int]:
        raw = self.pairs["test"]["relevant"].astype(int).tolist()
        return [int(v) for v in raw]


def build_job_grouped_context(
    semantic_splits: tuple[str, ...] = ("test",),
) -> JobGroupedContext:
    """Load the dataset and build the enriched, job_grouped context.

    Embeddings (MiniLM) are computed via the shared SemanticEmbedder so the
    semantic features are identical in construction to the 6E run.  Only the
    requested splits are embedded; the default (``test``) is all the primary
    benchmark error analysis needs.  Ablation retrains (tier 4) require no
    embeddings at all.
    """
    bundle = load_bundle()
    pairs = build_pairs(
        bundle.resumes,
        bundle.matches,
        negatives_per_job=NEGATIVES_PER_JOB,
        seed=SEED,
    )
    enriched = _join_pair_data(pairs, bundle.resumes, bundle.jobs)

    job_assign = job_group_split(bundle.jobs, seed=SEED, sizes=(0.7, 0.1, 0.2))
    manifest = SplitManifest(jobs=job_assign, resumes={})
    train_p, val_p, test_p = _split_pairs(enriched, manifest, "job_grouped")
    split_pairs = {"train": train_p, "val": val_p, "test": test_p}

    resume_vec = ResumeTextVectorizer(max_features=2000, min_df=2)
    job_vec = JobTextVectorizer(max_features=2000, min_df=2)
    fitted_resume_vec = resume_vec.fit(train_p)
    fitted_job_vec = job_vec.fit(train_p)

    embedder = SemanticEmbedder(batch_size=128)
    semantic_features: dict[str, pd.DataFrame] = {}
    for split_name in semantic_splits:
        split_pairs_df = split_pairs[split_name]
        emb = embedder.embed_pairs(split_pairs_df)
        semantic_features[split_name] = build_semantic_features(split_pairs_df, emb)
        # Attach the cosine to the pairs frame so error analysis heuristics
        # (pattern_counts) can inspect it without touching raw text.
        split_pairs[split_name]["minilm_cosine"] = emb.cosine

    return JobGroupedContext(
        pairs=split_pairs,
        fitted_resume_vec=fitted_resume_vec,
        fitted_job_vec=fitted_job_vec,
        semantic_features=semantic_features,
    )


def build_tier_features(
    ctx: JobGroupedContext,
    tier: int,
    pair_split: str = "test",
    excluded: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Build the feature matrix for a tier on one split, minus exclusions."""
    pairs = ctx.pairs[pair_split]
    X, metadata = _build_features_for_tier(
        tier,
        pairs,
        fitted_resume_vec=ctx.fitted_resume_vec,
        fitted_job_vec=ctx.fitted_job_vec,
        semantic_emb=None,
        semantic_features=ctx.semantic_features.get(pair_split),
    )
    if excluded:
        drop = [name for name in excluded if name in X.columns]
        if len(drop) != len(excluded):
            missing = set(excluded) - set(X.columns)
            raise ValueError(f"excluded feature not present: {missing}")
        X = X.drop(columns=drop)
    return X


def load_model_estimator(tier: int, split_type: str = "job_grouped") -> Any:
    """Load the fitted estimator saved by the 6E run for a tier/split.

    Tier 0 is a trivial majority-classifier that was not successfully
    persisted (nested class).  We reconstruct it here from the training
    labels, which is deterministic given the fixed split.
    """
    if tier == 0:
        return _rebuild_majority_classifier()
    path = PHASE_6E_DIR / f"model_tier{tier}_{split_type}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"model artifact not found: {path}")
    return joblib.load(path)


def _rebuild_majority_classifier() -> Any:
    """Reconstruct the tier-0 majority classifier from the training split.

    The training split is deterministic (seed=42, job_grouped) so this
    produces the exact same classifier that 6E trained.
    """
    from ml.models.trainer import _build_majority_classifier

    bundle = load_bundle()
    pairs = build_pairs(
        bundle.resumes, bundle.matches,
        negatives_per_job=NEGATIVES_PER_JOB, seed=SEED,
    )
    enriched = _join_pair_data(pairs, bundle.resumes, bundle.jobs)
    job_assign = job_group_split(
        bundle.jobs, seed=SEED, sizes=(0.7, 0.1, 0.2)
    )
    manifest = SplitManifest(jobs=job_assign, resumes={})
    train_p, _, _ = _split_pairs(enriched, manifest, "job_grouped")
    y_train = train_p["relevant"].astype(int).tolist()

    clf = _build_majority_classifier()
    clf.fit(np.zeros((len(y_train), 0)), np.asarray(y_train))
    return clf


def tier_feature_pipeline(
    ctx: JobGroupedContext,
    tier: int,
    excluded: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, list[int]]:
    """Return (X_test_awa_features, y_test) for analysis and reproduction."""
    X = build_tier_features(ctx, tier, "test", excluded)
    y = ctx.y_test()
    return X, y


def analyze_predictions_for_tier(
    ctx: JobGroupedContext,
    tier: int,
    excluded: tuple[str, ...] = (),
) -> tuple[list[int], pd.DataFrame]:
    """Produce test-set predictions for a tier using the 6E estimator.

    Unlike training/eval (which clone the estimator) this returns raw
    predictions from the saved model so error analysis can examine them
    alongside the ground-truth labels.
    """
    estimator = load_model_estimator(tier, "job_grouped")
    X, y = tier_feature_pipeline(ctx, tier, excluded)
    preds = estimator.predict(X.values.astype(float))
    return [int(p) for p in preds], X


def reproduce_tier_metrics(
    ctx: JobGroupedContext,
    tier: int,
) -> dict[str, float]:
    """Re-run the saved estimator on test and compare to 6E recorded metrics."""
    estimator = load_model_estimator(tier, "job_grouped")
    X, y_list = tier_feature_pipeline(ctx, tier)
    y_arr: np.ndarray = np.asarray(y_list, dtype=int)
    report = default_evaluator().evaluate(
        estimator, X, y_arr, tier=tier,
        tier_name=f"reproduce_tier{tier}",
        split_name=f"job_grouped_test_reproduce_{tier}",
    )
    return report.metrics


def train_ablation(
    ctx: JobGroupedContext,
    config: ExperimentConfig,
    excluded: tuple[str, ...],
) -> tuple[Any, list[str]]:
    """Train a tier-4-style logistic head on job_grouped train with exclusions.

    Returns ``(model, feature_names)``.  The caller evaluates on the frozen
    job_grouped test split; no test information reaches the training step.
    """
    from ml.experiments.config import TIERS as TIERS_REGISTRY

    X_train = build_tier_features(ctx, 4, "train", excluded)
    y_train_raw: list[int] = ctx.y_train()
    y_train: np.ndarray = np.asarray(y_train_raw, dtype=int)

    tier_spec = next(t for t in TIERS_REGISTRY if t.tier == 4)
    plan = ExperimentPlan(
        config=config, tier=4, model_spec=tier_spec.model,
        manifest=SplitManifest(jobs={}, resumes={}),
    )
    model = train(plan, X_train, y_train)
    return model, list(X_train.columns)
