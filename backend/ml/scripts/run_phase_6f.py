"""Phase 6F runner: model evaluation + error analysis + shortcut investigation.

Analysis-only.  Loads the 6E artifacts (no retraining of the benchmark tiers
except the explicitly controlled ablation retrains), reproduces job_grouped
test predictions for every tier, and produces the Phase 6F artifacts:

* feature_importance.json + feature_importance_tier5.json
* ablation_results.csv + ablation_metrics.json
* split_generalization.csv
* error_summary.json + confusion_tierN_job_grouped.json
* model_comparison.json
* sanity_checks.json
* robustness_results.json
* deterministic_baseline_analysis.json
* class_role_analysis.json

Run:
    cd backend && python -m ml.scripts.run_phase_6f
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

import numpy as np
import pandas as pd

from ml.deterministic_baseline.adapter import compute_phase5a_scores
from ml.evaluation.evaluator import default_evaluator
from ml.experiments.config import ExperimentConfig
from ml.loader import load_bundle
from ml.phase_6f.calibration import reliability_curve
from ml.phase_6f.context import (
    PHASE_6E_DIR,
    PHASE_6F_DIR,
    JobGroupedContext,
    analyze_predictions_for_tier,
    build_job_grouped_context,
    build_tier_features,
    load_model_estimator,
    reproduce_tier_metrics,
    train_ablation,
)
from ml.phase_6f.errors import (
    categorize,
    error_counts,
    pattern_counts,
    representative_examples,
)
from ml.phase_6f.importance import extract_importances, importance_by_family
from ml.phase_6f.robustness import (
    TRANSFORMS,
    apply_transform_to_frame,
    probe_prediction_stability,
)
from ml.phase_6f.sanity import (
    check_no_forbidden_columns,
    check_no_identity_features,
    check_quasi_generative_excluded,
    check_reproducible_metrics,
    check_split_disjointness,
    check_vectorizer_fit_on_train_only,
)
from ml.phase_6f.specs import ablation_specs
from ml.scripts.run_phase_6e import _build_features_for_tier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("phase_6f")

SEED = 42
ROBUSTNESS_SAMPLE = 500


def _ensure_dirs() -> None:
    PHASE_6F_DIR.mkdir(parents=True, exist_ok=True)


def _dump(name: str, payload: Any) -> None:
    path = PHASE_6F_DIR / name
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("saved %s", path.name)


def _read_ise_json(name: str) -> dict[str, Any]:
    path = PHASE_6E_DIR / name
    with open(path) as f:
        payload: dict[str, Any] = json.load(f)
    return payload


def _feature_names(ctx: JobGroupedContext, tier: int) -> list[str]:
    """Feature column names for a tier, aligned to the 6E training order."""
    X, _ = _build_features_for_tier(
        tier, ctx.pairs["test"],
        fitted_resume_vec=ctx.fitted_resume_vec,
        fitted_job_vec=ctx.fitted_job_vec,
        semantic_emb=None,
        semantic_features=ctx.semantic_features.get("test"),
    )
    return list(X.columns)


def _feature_importance(
    ctx: JobGroupedContext, tier: int, feature_names: list[str]
) -> dict[str, Any]:
    """Export ranked importances and family aggregates for a tier."""
    estimator = load_model_estimator(tier, "job_grouped")
    importances = extract_importances(estimator, feature_names)
    return {
        "tier": tier,
        "model_type": str(type(estimator).__name__),
        "n_features": len(feature_names),
        "ranked_head": importances[:50],
        "ranked_all": importances,
        "families": importance_by_family(importances),
    }


def _run_ablations(
    ctx: JobGroupedContext,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = ExperimentConfig.default()
    evaluator = default_evaluator()
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for spec in ablation_specs():
        excluded = spec.excluded
        X_test = build_tier_features(ctx, 4, "test", excluded)
        y_test = ctx.y_test()

        model, names = train_ablation(ctx, config, excluded)
        y_arr: np.ndarray = np.asarray(y_test, dtype=int)
        report = evaluator.evaluate(
            model, X_test, y_arr,
            tier=4, tier_name=f"ablation:{spec.id}",
            split_name="job_grouped_test",
        )
        m = report.metrics
        rows.append({
            "experiment_id": spec.id,
            "description": spec.description,
            "n_features": len(names),
            "excluded_features": "; ".join(excluded) or "none",
            "test_macro_f1": round(m["macro_f1"], 4),
            "test_accuracy": round(m["accuracy"], 4),
            "test_balanced_accuracy": round(m["balanced_accuracy"], 4),
        })
        details.append({
            "experiment_id": spec.id,
            "description": spec.description,
            "n_features": len(names),
            "excluded_features": list(excluded),
            "metrics": {k: round(v, 6) for k, v in m.items()},
            "confusion": report.confusion,
            "per_class": report.per_class,
        })
        logger.info(
            "ablation %s: macro_f1=%.4f acc=%.4f (n_feat=%d)",
            spec.id, m["macro_f1"], m["accuracy"], len(names),
        )
    return rows, details


def _deterministic_baseline_analysis(
    ctx: JobGroupedContext,
) -> dict[str, Any]:
    bundle = load_bundle()
    resumes_lookup = bundle.resumes.set_index("resume_id").to_dict("index")
    jobs_lookup = bundle.jobs.set_index("job_id").to_dict("index")

    test_pairs = ctx.pairs["test"]
    y_test = ctx.y_test()

    scores = compute_phase5a_scores(test_pairs, resumes_lookup, jobs_lookup)
    score_arr = np.array([0.0 if s is None else s for s in scores])

    det = build_tier_features(ctx, 4, "test")
    overlap_required = det["keyword_overlap_required"].to_numpy(dtype=float)
    overlap_preferred = det["keyword_overlap_preferred"].to_numpy(dtype=float)
    hybrid = det["hybrid_score"].to_numpy(dtype=float)

    label_arr = np.array(y_test, dtype=float)

    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 2 or b.size < 2:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    pos = label_arr == 1.0
    neg = label_arr == 0.0

    return {
        "correlation": {
            "score_vs_label": round(_pearson(score_arr, label_arr), 6),
            "score_vs_keyword_overlap_required": round(
                _pearson(score_arr, overlap_required), 6
            ),
            "score_vs_keyword_overlap_preferred": round(
                _pearson(score_arr, overlap_preferred), 6
            ),
            "score_vs_hybrid": round(_pearson(score_arr, hybrid), 6),
            "keyword_overlap_required_vs_label": round(
                _pearson(overlap_required, label_arr), 6
            ),
        },
        "score_distribution_by_label": {
            "positive_mean": round(float(score_arr[pos].mean()), 4),
            "positive_std": round(float(score_arr[pos].std()), 4),
            "negative_mean": round(float(score_arr[neg].mean()), 4),
            "negative_std": round(float(score_arr[neg].std()), 4),
        },
        "overlap_required_by_label": {
            "positive_mean": round(float(overlap_required[pos].mean()), 4),
            "negative_mean": round(float(overlap_required[neg].mean()), 4),
            "positive_ge_0_6": round(float((overlap_required[pos] >= 0.6).mean()), 4),
            "negative_ge_0_6": round(float((overlap_required[neg] >= 0.6).mean()), 4),
        },
        "n_test_pairs": len(y_test),
    }


def _bucket_summary(
    ctx: JobGroupedContext,
    y_pred: list[int],
    tier: int,
) -> dict[str, Any]:
    y_true = ctx.y_test()
    buckets = categorize(y_true, y_pred)
    summary: dict[str, Any] = {"tier": tier, "counts": error_counts(y_true, y_pred)}

    for name, label in (("tp", 1), ("tn", 0), ("fp", 0), ("fn", 1)):
        idxs = buckets[name]
        summary[name] = {
            "examples": representative_examples(ctx.pairs["test"], idxs, 5),
            "patterns": pattern_counts(ctx.pairs["test"], idxs, label=label),
        }
    return summary


def _model_comparison(
    ctx: JobGroupedContext,
    preds: dict[int, list[int]],
) -> dict[str, Any]:
    y_true = ctx.y_test()
    n = len(y_true)
    focal = (3, 4, 5)

    correct: dict[int, list[bool]] = {
        t: [p == self for self, p in zip(y_true, preds[t], strict=True)]
        for t in focal
    }

    agreement: dict[str, Any] = {}
    for a in focal:
        for b in focal:
            if a < b:
                same = sum(1 for x, y in zip(preds[a], preds[b], strict=True) if x == y)
                agreement[f"{a}_vs_{b}"] = {"same": same, "diff": n - same}

    divergence: dict[str, Any] = {}
    for a in focal:
        for b in focal:
            if a == b:
                continue
            only_a = sum(1 for ca, cb in zip(correct[a], correct[b], strict=True)
                         if ca and not cb)
            divergence[f"correct_{a}_wrong_{b}"] = only_a

    return {
        "n_test_pairs": n,
        "accuracy": {
            t: round(len([c for c in correct[t] if c]) / n, 6) for t in focal
        },
        "agreement": agreement,
        "divergence": divergence,
    }


def _sanity_checks(ctx: JobGroupedContext) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    X_test = build_tier_features(ctx, 4, "test")
    checks.append(check_no_forbidden_columns(X_test))
    checks.append(check_no_identity_features(X_test))

    for spec in ablation_specs():
        X = build_tier_features(ctx, 4, "test", spec.excluded)
        checks.append(check_quasi_generative_excluded(X, spec.excluded))

    bundle = load_bundle()
    from ml.data.pairs import build_pairs
    all_pairs = build_pairs(
        bundle.resumes, bundle.matches, negatives_per_job=30, seed=SEED
    )
    jg_splits = {
        "train": len(ctx.pairs["train"]),
        "val": len(ctx.pairs["val"]),
        "test": len(ctx.pairs["test"]),
    }
    checks.append({
        "check": "job_grouped_row_budget_matches_dataset",
        "passed": sum(jg_splits.values()) == len(all_pairs),
        "detail": f"pairs_total={len(all_pairs)}, splits={jg_splits}",
    })
    if not sum(jg_splits.values()) == len(all_pairs):
        raise AssertionError("job_grouped split does not cover every pair row")

    from ml.splits import resume_group_split, strict_both_split
    sb = strict_both_split(bundle.resumes, bundle.jobs, bundle.matches, seed=SEED)
    sb_jobs = _split_members(sb.jobs)
    sb_resumes = _split_members(sb.resumes)
    checks.append(check_split_disjointness(
        sb_jobs["train"], sb_jobs["val"], sb_jobs["test"], "strict_both_jobs"))
    checks.append(
        check_split_disjointness(
            sb_resumes["train"], sb_resumes["val"], sb_resumes["test"],
            "strict_both_resumes",
        )
    )

    rg = resume_group_split(bundle.resumes, seed=SEED, sizes=(0.7, 0.1, 0.2))
    rg_members = _split_members(rg)
    checks.append(
        check_split_disjointness(
            rg_members["train"], rg_members["val"], rg_members["test"],
            "resume_grouped_resumes",
        )
    )

    train_tokens = _token_set_from_pairs(ctx.pairs["train"])
    test_tokens = _token_set_from_pairs(ctx.pairs["test"])
    vocab = _fitted_vocab_size(ctx)
    checks.append(check_vectorizer_fit_on_train_only(vocab, train_tokens, test_tokens))

    for tier in (0, 1, 2, 3, 4, 5):
        recorded = _read_ise_json(
            f"metrics_tier{tier}_job_grouped.json"
        )["test_metrics"]
        reproduced = reproduce_tier_metrics(ctx, tier)
        recorded_core = {
            k: float(recorded[k])
            for k in ("macro_f1", "accuracy", "balanced_accuracy")
        }
        checks.append(
            check_reproducible_metrics(recorded_core, reproduced, tolerance=1e-4)
        )

    return {"note": "assertions stop the run if a check fails", "checks": checks}


def _split_members(assignment: dict[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for key, split in assignment.items():
        if split in out:
            out[split].append(key)
    return out


def _token_set_from_pairs(pairs: pd.DataFrame) -> set[str]:
    tokens: set[str] = set()
    for col in ("summary", "experience_bullets", "skills"):
        values = pairs.get(col)
        if values is None:
            continue
        for val in values.tolist():
            if val is None:
                continue
            if isinstance(val, (list, tuple)):
                for part in val:
                    tokens.update(str(part).lower().split())
            else:
                tokens.update(str(val).lower().split())
    return tokens


def _fitted_vocab_size(ctx: JobGroupedContext) -> int:
    return int(
        len(ctx.fitted_resume_vec.vectorizer.vocabulary_)
        + len(ctx.fitted_job_vec.vectorizer.vocabulary_)
    )


def _robustness_probe(ctx: JobGroupedContext) -> dict[str, Any]:
    rng = random.Random(SEED)
    sample_rows = rng.sample(range(len(ctx.pairs["test"])), ROBUSTNESS_SAMPLE)
    sample_pairs = ctx.pairs["test"].iloc[sample_rows].copy()

    estimator = load_model_estimator(4, "job_grouped")

    def _predict(pairs: pd.DataFrame) -> list[int]:
        X, _ = _build_features_for_tier(
            4, pairs,
            fitted_resume_vec=None, fitted_job_vec=None,
            semantic_emb=None, semantic_features=None,
        )
        return [int(p) for p in estimator.predict(X.values.astype(float))]

    X_base, _ = _build_features_for_tier(
        4, sample_pairs,
        fitted_resume_vec=None, fitted_job_vec=None,
        semantic_emb=None, semantic_features=None,
    )
    y_base = [int(p) for p in estimator.predict(X_base.values.astype(float))]

    results: list[dict[str, Any]] = []
    for name, fn in TRANSFORMS:
        transformed = apply_transform_to_frame(sample_pairs, fn)
        X_probe, _ = _build_features_for_tier(
            4, transformed,
            fitted_resume_vec=None, fitted_job_vec=None,
            semantic_emb=None, semantic_features=None,
        )
        y_probe = [int(p) for p in estimator.predict(X_probe.values.astype(float))]
        results.append({
            "transform": name,
            "result": probe_prediction_stability(y_base, y_probe, X_base, X_probe),
        })
    return {"sample_size": ROBUSTNESS_SAMPLE, "transforms": results}


def _class_role_analysis(
    ctx: JobGroupedContext, preds: dict[int, list[int]]
) -> dict[str, Any]:
    pairs = ctx.pairs["test"]
    roles = pairs["job_title"].astype(str).to_numpy()
    y_true = np.array(ctx.y_test())
    frames: dict[str, Any] = {}
    for tier, pred in preds.items():
        y_pred = np.array(pred)
        accuracies: dict[str, Any] = {}
        sizes: dict[str, Any] = {}
        for role in sorted(set(roles.tolist())):
            mask = roles == role
            n = int(mask.sum())
            if n < 30:
                continue
            accuracies[role] = round(float((y_true[mask] == y_pred[mask]).mean()), 6)
            sizes[role] = n
        frames[f"tier{tier}"] = {"accuracy": accuracies, "n_cases": sizes}
    return frames


def _split_generalization_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tier in (0, 1, 2, 3, 4, 5):
        row: dict[str, Any] = {"tier": tier}
        row["tier_name"] = _read_ise_json(
            f"metrics_tier{tier}_job_grouped.json"
        )["tier_name"]
        for split in ("job_grouped", "strict_both", "resume_grouped"):
            metrics = _read_ise_json(f"metrics_tier{tier}_{split}.json")
            row[f"{split}_macro_f1"] = round(metrics["test_metrics"]["macro_f1"], 4)
            row[f"{split}_accuracy"] = round(metrics["test_metrics"]["accuracy"], 4)
        rows.append(row)
    return rows


def _calibration_analysis(
    ctx: JobGroupedContext,
) -> dict[str, Any]:
    """Reliability curves for probabilistic tiers on job_grouped test."""
    y_true = ctx.y_test()
    out: dict[str, Any] = {}
    for tier in (3, 4, 5):
        estimator = load_model_estimator(tier, "job_grouped")
        if not hasattr(estimator, "predict_proba"):
            out[f"tier{tier}"] = {"skipped": "no predict_proba"}
            continue
        X = build_tier_features(ctx, tier, "test")
        probs = list(
            estimator.predict_proba(X.values.astype(float))[:, 1]
        )
        out[f"tier{tier}"] = reliability_curve(
            y_true, probs, bins=10
        )
    return out


def run_all() -> dict[str, Any]:
    _ensure_dirs()
    logger.info("building job_grouped context (embeds test split only)...")
    ctx = build_job_grouped_context()

    logger.info("reproducing test predictions for tiers 0-5...")
    preds: dict[int, list[int]] = {}
    for tier in (0, 1, 2, 3, 4, 5):
        preds[tier], _ = analyze_predictions_for_tier(ctx, tier)
        logger.info("tier %d: %d predictions", tier, len(preds[tier]))

    logger.info("feature importance tier 4...")
    feat4_names = _feature_names(ctx, 4)
    feat4 = _feature_importance(ctx, 4, feat4_names)
    logger.info("feature importance tier 5...")
    feat5_names = _feature_names(ctx, 5)
    feat5 = _feature_importance(ctx, 5, feat5_names)

    logger.info("running ablations...")
    ablation_rows, ablation_details = _run_ablations(ctx)
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_path = PHASE_6F_DIR / "ablation_results.csv"
    ablation_df.to_csv(ablation_path, index=False)
    logger.info("saved %s", ablation_path.name)

    logger.info("deterministic baseline analysis...")
    det_analysis = _deterministic_baseline_analysis(ctx)

    logger.info("error summaries (tiers 3,4,5)...")
    error_summary = {
        "split": "job_grouped_test",
        "tier3": _bucket_summary(ctx, preds[3], 3),
        "tier4": _bucket_summary(ctx, preds[4], 4),
        "tier5": _bucket_summary(ctx, preds[5], 5),
    }

    logger.info("model comparison...")
    model_comparison = _model_comparison(ctx, preds)

    logger.info("sanity checks...")
    sanity = _sanity_checks(ctx)

    logger.info("robustness probe...")
    robustness = _robustness_probe(ctx)

    logger.info("class/role analysis...")
    class_role = _class_role_analysis(ctx, preds)

    logger.info("calibration (tiers 3,4,5)...")
    calibration = _calibration_analysis(ctx)

    split_generalization = _split_generalization_table()

    _dump("feature_importance.json", feat4)
    _dump("feature_importance_tier5.json", feat5)
    _dump("ablation_metrics.json", ablation_details)
    _dump("deterministic_baseline_analysis.json", det_analysis)
    _dump("error_summary.json", error_summary)
    _dump("model_comparison.json", model_comparison)
    _dump("sanity_checks.json", sanity)
    _dump("robustness_results.json", robustness)
    _dump("class_role_analysis.json", class_role)
    _dump("calibration_analysis.json", calibration)
    _dump("confusion_tier3_job_grouped.json",
          {"confusion": error_counts(ctx.y_test(), preds[3])["confusion"]})
    _dump("confusion_tier4_job_grouped.json",
          {"confusion": error_counts(ctx.y_test(), preds[4])["confusion"]})
    _dump("confusion_tier5_job_grouped.json",
          {"confusion": error_counts(ctx.y_test(), preds[5])["confusion"]})

    sg_df = pd.DataFrame(split_generalization)
    sg_path = PHASE_6F_DIR / "split_generalization.csv"
    sg_df.to_csv(sg_path, index=False)
    logger.info("saved %s", sg_path.name)

    summary = {
        "ablation_rows": ablation_rows,
        "config_fingerprint": ExperimentConfig.default().fingerprint(),
    }
    _dump("experiment_summary.json", summary)

    logger.info("=" * 80)
    logger.info("PHASE 6F SUMMARY (job_grouped test)")
    logger.info("%-12s %-10s %-10s %-10s", "ablation", "n_feat", "macro_f1", "acc")
    for row in ablation_rows:
        logger.info(
            "%-12s %-10d %-10.4f %-10.4f",
            row["experiment_id"], row["n_features"], row["test_macro_f1"],
            row["test_accuracy"],
        )
    return summary


if __name__ == "__main__":
    run_all()
