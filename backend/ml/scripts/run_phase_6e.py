"""Phase 6E experiment runner.

Loads the audited dataset, builds pairs/splits, trains all experiment tiers,
evaluates them, saves artifacts, and produces the comparison table.  Run with:

    cd backend && python -m ml.scripts.run_phase_6e
"""

from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.config import ARTIFACTS_DIR, DATASET_ID, DATASET_REVISION
from ml.data.pairs import build_pairs, label_counts
from ml.evaluation.evaluator import default_evaluator
from ml.experiments.config import (
    TIERS,
    ExperimentConfig,
)
from ml.features.ats_features import build_ats_6a_features, build_ats_6b_features
from ml.features.deterministic import build_deterministic_match_features
from ml.features.semantic import SemanticEmbedder, build_semantic_features
from ml.features.structured_features import build_structured_features
from ml.features.text import JobTextVectorizer, ResumeTextVectorizer
from ml.loader import load_bundle
from ml.models.trainer import ExperimentPlan, train
from ml.splits import (
    SplitManifest,
    job_group_split,
    resume_group_split,
    strict_both_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("phase_6e")

ARTIFACT_DIR = ARTIFACTS_DIR / "phase_6e"
SEED = 42
NEGATIVES_PER_JOB = 30


def _ensure_dirs() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Loading dataset bundle...")
    bundle = load_bundle()
    logger.info(
        "Loaded: %d resumes, %d jobs, %d matches",
        len(bundle.resumes), len(bundle.jobs), len(bundle.matches),
    )
    return bundle.resumes, bundle.jobs, bundle.matches


def _get_job_columns(jobs: pd.DataFrame) -> pd.DataFrame:
    return jobs[["job_id", "job_title", "seniority", "industry"]].copy()


def _get_resume_columns(resumes: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "resume_id", "role", "seniority",
        "years_experience", "industry", "education",
    ]
    present = [c for c in cols if c in resumes.columns]
    return resumes[present].copy()


def _join_pair_data(
    pairs: pd.DataFrame,
    resumes: pd.DataFrame,
    jobs: pd.DataFrame,
) -> pd.DataFrame:
    resume_cols = _get_resume_columns(resumes)
    job_cols = _get_job_columns(jobs)

    resume_lookup = resume_cols.set_index("resume_id").to_dict("index")
    job_lookup = job_cols.set_index("job_id").to_dict("index")

    enriched = pairs.copy()

    resume_features = []
    for _, row in enriched.iterrows():
        rid = str(row["resume_id"])
        info = resume_lookup.get(rid, {})
        resume_features.append({
            "resume_role": info.get("role"),
            "resume_seniority": info.get("seniority"),
            "resume_years_experience": info.get("years_experience"),
            "resume_industry": info.get("industry"),
            "resume_education": info.get("education"),
        })
    resume_df = pd.DataFrame(resume_features, index=enriched.index)

    job_features = []
    for _, row in enriched.iterrows():
        jid = str(row["job_id"])
        info = job_lookup.get(jid, {})
        job_features.append({
            "job_title": info.get("job_title"),
            "job_seniority": info.get("seniority"),
            "job_industry": info.get("industry"),
        })
    job_df = pd.DataFrame(job_features, index=enriched.index)

    enriched = pd.concat([enriched, resume_df, job_df], axis=1)

    for col in ("summary", "experience_bullets", "skills"):
        if col in resumes.columns:
            lookup = resumes.set_index("resume_id")[col].to_dict()
            enriched[col] = enriched["resume_id"].map(lookup)

    for col in ("description", "responsibilities", "requirements"):
        if col in jobs.columns:
            lookup = jobs.set_index("job_id")[col].to_dict()
            enriched[col] = enriched["job_id"].map(lookup)

    if "must_have_skills" in jobs.columns:
        lookup = jobs.set_index("job_id")["must_have_skills"].to_dict()
        enriched["job_must_have_skills"] = enriched["job_id"].map(lookup)

    if "nice_to_have_skills" in jobs.columns:
        lookup = jobs.set_index("job_id")["nice_to_have_skills"].to_dict()
        enriched["job_nice_to_have_skills"] = enriched["job_id"].map(lookup)

    if "skills" in resumes.columns:
        enriched["resume_skills"] = enriched["skills"]

    return enriched


def _strict_label(row: pd.Series) -> str:
    if row["_job_split"] == "test" or row["_resume_split"] == "test":
        return "test"
    if row["_resume_split"] == "val":
        return "val"
    return str(row["_job_split"])


def _split_pairs(
    pairs: pd.DataFrame,
    manifest: SplitManifest,
    split_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split_type == "job_grouped":
        job_split = manifest.jobs
        pairs = pairs.copy()
        pairs["_split"] = pairs["job_id"].map(job_split)
    elif split_type == "resume_grouped":
        resume_split = manifest.resumes
        pairs = pairs.copy()
        pairs["_split"] = pairs["resume_id"].map(resume_split).fillna("train")
    elif split_type == "strict_both":
        job_split = manifest.jobs
        resume_split = manifest.resumes
        pairs = pairs.copy()
        pairs["_job_split"] = pairs["job_id"].map(job_split)
        pairs["_resume_split"] = pairs["resume_id"].map(resume_split).fillna("train")
        pairs["_split"] = pairs.apply(
            lambda r: _strict_label(r),
            axis=1,
        )
    else:
        raise ValueError(f"unknown split_type: {split_type}")

    train_pairs = pairs[pairs["_split"] == "train"].drop(columns=["_split"])
    val_pairs = pairs[pairs["_split"] == "val"].drop(columns=["_split"])
    test_pairs = pairs[pairs["_split"] == "test"].drop(columns=["_split"])
    return train_pairs, val_pairs, test_pairs


def _build_features_for_tier(
    tier: int,
    pairs: pd.DataFrame,
    resume_vec: ResumeTextVectorizer | None = None,
    job_vec: JobTextVectorizer | None = None,
    fitted_resume_vec: Any = None,
    fitted_job_vec: Any = None,
    semantic_emb: SemanticEmbedder | None = None,
    semantic_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build feature matrix for a given tier. Returns (X, metadata)."""
    from ml.experiments.config import TIERS as _TIERS

    tier_spec = next(t for t in _TIERS if t.tier == tier)
    families = tier_spec.feature_families
    metadata: dict[str, Any] = {"feature_families": list(families)}

    parts: list[pd.DataFrame] = []

    if "resume_text" in families and fitted_resume_vec is not None:
        resume_feats = fitted_resume_vec.transform(pairs)
        parts.append(resume_feats)

    if "job_text" in families and fitted_job_vec is not None:
        job_feats = fitted_job_vec.transform(pairs)
        parts.append(job_feats)

    if "minilm" in families and semantic_features is not None:
        parts.append(semantic_features)

    if "structured_resume" in families or "structured_job" in families:
        struct_feats = build_structured_features(pairs)
        parts.append(struct_feats)

    if "deterministic_match" in families:
        det_feats = build_deterministic_match_features(pairs)
        parts.append(det_feats)

    if "ats_6a" in families:
        ats6a_feats = build_ats_6a_features(pairs)
        parts.append(ats6a_feats)

    if "ats_6b" in families:
        ats6b_feats = build_ats_6b_features(pairs)
        parts.append(ats6b_feats)

    if not parts:
        return pd.DataFrame(index=pairs.index), metadata

    X = pd.concat(parts, axis=1)
    X = X.fillna(0.0)
    metadata["n_features"] = len(X.columns)
    metadata["feature_names"] = list(X.columns)
    return X, metadata


def run_experiment(
    config: ExperimentConfig,
    tier: int,
    manifest: SplitManifest,
    split_type: str,
    pairs_full: pd.DataFrame,
    resumes: pd.DataFrame,
    jobs: pd.DataFrame,
    resume_vec: ResumeTextVectorizer | None = None,
    job_vec: JobTextVectorizer | None = None,
    fitted_resume_vec: Any = None,
    fitted_job_vec: Any = None,
    semantic_emb: SemanticEmbedder | None = None,
    semantic_features_train: pd.DataFrame | None = None,
    semantic_features_val: pd.DataFrame | None = None,
    semantic_features_test: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run one experiment tier on one split type. Returns results dict."""
    tier_spec = next(t for t in TIERS if t.tier == tier)
    plan = ExperimentPlan(
        config=config, tier=tier, model_spec=tier_spec.model, manifest=manifest
    )

    train_pairs, val_pairs, test_pairs = _split_pairs(pairs_full, manifest, split_type)
    y_train = train_pairs["relevant"].values.astype(int)
    y_val = val_pairs["relevant"].values.astype(int)
    y_test = test_pairs["relevant"].values.astype(int)

    logger.info(
        "Tier %d (%s) on %s: train=%d, val=%d, test=%d",
        tier, tier_spec.name, split_type,
        len(train_pairs), len(val_pairs), len(test_pairs),
    )

    X_train, _ = _build_features_for_tier(
        tier, train_pairs, resume_vec, job_vec, fitted_resume_vec, fitted_job_vec,
        semantic_emb, semantic_features_train,
    )
    X_val, _ = _build_features_for_tier(
        tier, val_pairs, resume_vec, job_vec, fitted_resume_vec, fitted_job_vec,
        semantic_emb, semantic_features_val,
    )
    X_test, _ = _build_features_for_tier(
        tier, test_pairs, resume_vec, job_vec, fitted_resume_vec, fitted_job_vec,
        semantic_emb, semantic_features_test,
    )

    if tier == 0 and len(X_train.columns) == 0:
        X_train = pd.DataFrame(
            {"_placeholder": np.zeros(len(train_pairs))}, index=train_pairs.index
        )
        X_val = pd.DataFrame(
            {"_placeholder": np.zeros(len(val_pairs))}, index=val_pairs.index
        )
        X_test = pd.DataFrame(
            {"_placeholder": np.zeros(len(test_pairs))}, index=test_pairs.index
        )

    model = train(plan, X_train, y_train)

    evaluator = default_evaluator()
    test_report = evaluator.evaluate(
        model,
        X_test,
        y_test,
        tier=tier,
        tier_name=tier_spec.name,
        split_name=f"{split_type}_test",
    )
    val_report = evaluator.evaluate(
        model,
        X_val,
        y_val,
        tier=tier,
        tier_name=tier_spec.name,
        split_name=f"{split_type}_val",
    )

    return {
        "tier": tier,
        "tier_name": tier_spec.name,
        "split_type": split_type,
        "model": model,
        "train_samples": len(train_pairs),
        "val_samples": len(val_pairs),
        "test_samples": len(test_pairs),
        "n_features": len(X_train.columns),
        "train_time_s": model.train_time_s,
        "test_report": test_report,
        "val_report": val_report,
        "feature_names": list(X_train.columns),
    }


def save_artifacts(
    results: list[dict[str, Any]],
    config: ExperimentConfig,
    dataset_meta: dict[str, Any],
    split_meta: dict[str, Any],
    semantic_meta: dict[str, Any] | None,
) -> None:
    """Save experiment artifacts to the phase-6e directory."""
    _ensure_dirs()

    config_path = ARTIFACT_DIR / "experiment_config.json"
    with open(config_path, "w") as f:
        json.dump(config.model_dump(mode="json"), f, indent=2, default=str)
    logger.info("Saved experiment config to %s", config_path)

    dataset_path = ARTIFACT_DIR / "dataset_metadata.json"
    with open(dataset_path, "w") as f:
        json.dump(dataset_meta, f, indent=2, default=str)
    logger.info("Saved dataset metadata to %s", dataset_path)

    split_path = ARTIFACT_DIR / "split_metadata.json"
    with open(split_path, "w") as f:
        json.dump(split_meta, f, indent=2, default=str)
    logger.info("Saved split metadata to %s", split_path)

    if semantic_meta:
        sem_path = ARTIFACT_DIR / "semantic_metadata.json"
        with open(sem_path, "w") as f:
            json.dump(semantic_meta, f, indent=2, default=str)
        logger.info("Saved semantic metadata to %s", sem_path)

    comparison_rows = []
    for r in results:
        test_m = r["test_report"].metrics
        val_m = r["val_report"].metrics
        confusion = r["test_report"].confusion
        comparison_rows.append({
            "tier": r["tier"],
            "tier_name": r["tier_name"],
            "split_type": r["split_type"],
            "train_samples": r["train_samples"],
            "test_samples": r["test_samples"],
            "n_features": r["n_features"],
            "train_time_s": round(r["train_time_s"], 3),
            "val_macro_f1": round(val_m.get("macro_f1", 0.0), 4),
            "test_macro_f1": round(test_m.get("macro_f1", 0.0), 4),
            "test_accuracy": round(test_m.get("accuracy", 0.0), 4),
            "test_balanced_accuracy": round(test_m.get("balanced_accuracy", 0.0), 4),
            "test_pos_precision": round(test_m.get("positive_precision", 0.0), 4),
            "test_pos_recall": round(test_m.get("positive_recall", 0.0), 4),
            "test_pos_f1": round(test_m.get("positive_f1", 0.0), 4),
            "test_neg_precision": round(test_m.get("negative_precision", 0.0), 4),
            "test_neg_recall": round(test_m.get("negative_recall", 0.0), 4),
            "test_neg_f1": round(test_m.get("negative_f1", 0.0), 4),
            "confusion_tp": confusion["tp"],
            "confusion_fp": confusion["fp"],
            "confusion_fn": confusion["fn"],
            "confusion_tn": confusion["tn"],
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_path = ARTIFACT_DIR / "comparison_table.csv"
    comp_df.to_csv(comp_path, index=False)
    logger.info("Saved comparison table to %s", comp_path)

    comp_json_path = ARTIFACT_DIR / "comparison_table.json"
    with open(comp_json_path, "w") as f:
        json.dump(comparison_rows, f, indent=2)
    logger.info("Saved comparison JSON to %s", comp_json_path)

    for r in results:
        tier = r["tier"]
        split = r["split_type"]
        tag = f"tier{tier}_{split}"
        try:
            model_path = ARTIFACT_DIR / f"model_{tag}.joblib"
            joblib.dump(r["model"].estimator, model_path)
            logger.info("Saved model for %s to %s", tag, model_path)
        except Exception:
            logger.warning(
                "Could not serialize model for %s; skipping binary", tag
            )

        metrics_path = ARTIFACT_DIR / f"metrics_{tag}.json"
        metrics_data = {
            "tier": tier,
            "tier_name": r["tier_name"],
            "split_type": split,
            "train_samples": r["train_samples"],
            "test_samples": r["test_samples"],
            "n_features": r["n_features"],
            "train_time_s": r["train_time_s"],
            "test_metrics": r["test_report"].metrics,
            "val_metrics": r["val_report"].metrics,
            "test_confusion": r["test_report"].confusion,
            "test_per_class": r["test_report"].per_class,
            "test_metadata": r["test_report"].metadata,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2, default=str)
        logger.info("Saved metrics for %s to %s", tag, metrics_path)

    summary_path = ARTIFACT_DIR / "experiment_summary.json"
    summary = {
        "config_fingerprint": config.fingerprint(),
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "seed": config.seed,
        "negatives_per_job": config.negatives_per_job,
        "total_experiments": len(results),
        "best_macro_f1": max(
            r["test_report"].metrics.get("macro_f1", 0.0) for r in results
        ),
        "experiments": [
            {
                "tier": r["tier"],
                "tier_name": r["tier_name"],
                "split_type": r["split_type"],
                "test_macro_f1": round(
                    r["test_report"].metrics.get("macro_f1", 0.0), 4
                ),
            }
            for r in results
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved experiment summary to %s", summary_path)


def run_all() -> list[dict[str, Any]]:
    """Run all experiment tiers across all split types."""
    config = ExperimentConfig.default()
    config.validate_tiers()

    _ensure_dirs()

    resumes, jobs, matches = _load_data()

    dataset_meta = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "n_resumes": len(resumes),
        "n_jobs": len(jobs),
        "n_matches": len(matches),
        "seed": SEED,
        "negatives_per_job": NEGATIVES_PER_JOB,
    }

    logger.info("Building pairs...")
    full_pairs = build_pairs(
        resumes, matches, negatives_per_job=NEGATIVES_PER_JOB, seed=SEED
    )
    pair_counts = label_counts(full_pairs)
    dataset_meta["pair_counts"] = pair_counts
    logger.info("Pair counts: %s", pair_counts)

    logger.info("Enriching pairs with resume/job data...")
    enriched_pairs = _join_pair_data(full_pairs, resumes, jobs)

    split_meta: dict[str, Any] = {}
    manifests: dict[str, SplitManifest] = {}

    logger.info("Computing job_grouped split...")
    job_assign = job_group_split(jobs, seed=SEED, sizes=(0.7, 0.1, 0.2))
    manifests["job_grouped"] = SplitManifest(jobs=job_assign, resumes={})

    logger.info("Computing strict_both split...")
    manifests["strict_both"] = strict_both_split(resumes, jobs, matches, seed=SEED)

    logger.info("Computing resume_grouped split...")
    resume_assign = resume_group_split(resumes, seed=SEED, sizes=(0.7, 0.1, 0.2))
    manifests["resume_grouped"] = SplitManifest(jobs={}, resumes=resume_assign)

    for stype, manifest in manifests.items():
        train_p, val_p, test_p = _split_pairs(enriched_pairs, manifest, stype)
        split_meta[stype] = {
            "train": len(train_p),
            "val": len(val_p),
            "test": len(test_p),
            "total": len(train_p) + len(val_p) + len(test_p),
            "job_groups_in_test": len(set(test_p["job_id"])) if len(test_p) > 0 else 0,
        }

    resume_vec = ResumeTextVectorizer(max_features=2000, min_df=2)
    job_vec = JobTextVectorizer(max_features=2000, min_df=2)

    job_group_manifest = manifests["job_grouped"]
    train_p_jg, _, _ = _split_pairs(enriched_pairs, job_group_manifest, "job_grouped")

    logger.info("Fitting TF-IDF vectorizers on job_grouped training split...")
    fitted_resume_vec = resume_vec.fit(train_p_jg)
    fitted_job_vec = job_vec.fit(train_p_jg)

    logger.info("Initializing MiniLM embedder...")
    semantic_emb = SemanticEmbedder(batch_size=128)

    semantic_meta = {
        "model_name": semantic_emb.model_name,
        "embedding_dim": 384,
        "batch_size": 128,
    }

    logger.info("Generating semantic embeddings for job_grouped splits...")
    train_p_jg, val_p_jg, test_p_jg = _split_pairs(
        enriched_pairs, job_group_manifest, "job_grouped"
    )
    emb_train_jg = semantic_emb.embed_pairs(train_p_jg)
    emb_val_jg = semantic_emb.embed_pairs(val_p_jg)
    emb_test_jg = semantic_emb.embed_pairs(test_p_jg)
    semantic_features_train_jg = build_semantic_features(train_p_jg, emb_train_jg)
    semantic_features_val_jg = build_semantic_features(val_p_jg, emb_val_jg)
    semantic_features_test_jg = build_semantic_features(test_p_jg, emb_test_jg)
    semantic_meta["device"] = emb_train_jg.device

    all_results: list[dict[str, Any]] = []

    for split_type in ["job_grouped", "strict_both", "resume_grouped"]:
        logger.info("=== Split: %s ===", split_type)
        manifest = manifests[split_type]

        if split_type == "job_grouped":
            emb_train = semantic_features_train_jg
            emb_val = semantic_features_val_jg
            emb_test = semantic_features_test_jg
        elif split_type == "strict_both":
            train_p_sb, val_p_sb, test_p_sb = _split_pairs(
                enriched_pairs, manifest, split_type
            )
            emb_train = build_semantic_features(
                train_p_sb, semantic_emb.embed_pairs(train_p_sb)
            )
            emb_val = build_semantic_features(
                val_p_sb, semantic_emb.embed_pairs(val_p_sb)
            )
            emb_test = build_semantic_features(
                test_p_sb, semantic_emb.embed_pairs(test_p_sb)
            )
        else:
            train_p_rg, val_p_rg, test_p_rg = _split_pairs(
                enriched_pairs, manifest, split_type
            )
            emb_train = build_semantic_features(
                train_p_rg, semantic_emb.embed_pairs(train_p_rg)
            )
            emb_val = build_semantic_features(
                val_p_rg, semantic_emb.embed_pairs(val_p_rg)
            )
            emb_test = build_semantic_features(
                test_p_rg, semantic_emb.embed_pairs(test_p_rg)
            )

        for tier in config.tiers:
            tier_spec = next(t for t in TIERS if t.tier == tier)
            logger.info("Running tier %d: %s", tier, tier_spec.name)

            try:
                result = run_experiment(
                    config=config,
                    tier=tier,
                    manifest=manifest,
                    split_type=split_type,
                    pairs_full=enriched_pairs,
                    resumes=resumes,
                    jobs=jobs,
                    fitted_resume_vec=fitted_resume_vec,
                    fitted_job_vec=fitted_job_vec,
                    semantic_emb=semantic_emb,
                    semantic_features_train=emb_train,
                    semantic_features_val=emb_val,
                    semantic_features_test=emb_test,
                )
                all_results.append(result)
                logger.info(
                    "  -> test_macro_f1=%.4f, accuracy=%.4f",
                    result["test_report"].metrics.get("macro_f1", 0.0),
                    result["test_report"].metrics.get("accuracy", 0.0),
                )
            except Exception:
                logger.exception("Tier %d on %s FAILED", tier, split_type)

    save_artifacts(all_results, config, dataset_meta, split_meta, semantic_meta)

    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(
        "%-6s %-42s %-16s %-10s %-10s %-10s %-10s",
        "Tier", "Name", "Split", "MF1", "Acc", "B-Acc", "Time(s)",
    )
    logger.info("-" * 80)
    for r in all_results:
        m = r["test_report"].metrics
        logger.info(
            "%-6d %-42s %-16s %-10.4f %-10.4f %-10.4f %-10.3f",
            r["tier"],
            r["tier_name"][:42],
            r["split_type"],
            m.get("macro_f1", 0.0),
            m.get("accuracy", 0.0),
            m.get("balanced_accuracy", 0.0),
            r["train_time_s"],
        )
    logger.info("=" * 80)

    return all_results


if __name__ == "__main__":
    run_all()
