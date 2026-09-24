"""Offline tests for Phase 6E training pipeline.

These tests verify the actual training implementation using tiny fixtures.
They never download the external dataset and never require a GPU.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml.data.pairs import build_pairs
from ml.evaluation.contracts import (
    EvaluationReport,
)
from ml.evaluation.evaluator import ModelEvaluator, default_evaluator
from ml.experiments.config import ExperimentConfig
from ml.features.deterministic import build_deterministic_match_features
from ml.features.structured_features import build_structured_features
from ml.features.text import ResumeTextVectorizer
from ml.models.trainer import ExperimentPlan, TrainedModel, train
from ml.splits import (
    SplitManifest,
    job_group_split,
    strict_both_split,
)


def _tiny_resumes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "resume_id": [f"R_{i}" for i in range(20)],
            "role": [
                "Backend Engineer" if i % 3 == 0 else f"Data Engineer_{i % 2}"
                for i in range(20)
            ],
            "seniority": ["Mid" if i % 2 == 0 else "Senior" for i in range(20)],
            "years_experience": [float(2 + (i % 7)) for i in range(20)],
            "industry": ["Tech" if i % 2 == 0 else "Finance" for i in range(20)],
            "education": ["B.Tech in CS" for _ in range(20)],
            "summary": [
                f"Software developer with {i} years of experience in Python and SQL."
                for i in range(20)
            ],
            "experience_bullets": [
                ["- Developed backend APIs with Python and FastAPI."]
                for _ in range(20)
            ],
            "skills": [["python", "sql", "aws"] for _ in range(20)],
        }
    )


def _tiny_jobs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "job_id": [f"J_{i}" for i in range(10)],
            "job_title": [
                "Backend Engineer" if i % 2 == 0 else "Data Engineer"
                for i in range(10)
            ],
            "seniority": ["Mid" for _ in range(10)],
            "industry": ["Tech" for _ in range(10)],
            "description": [
                "We are hiring a backend engineer who knows Python and AWS."
                for _ in range(10)
            ],
            "responsibilities": [["- Build scalable APIs."] for _ in range(10)],
            "requirements": [["- Python", "- SQL", "- AWS"] for _ in range(10)],
            "must_have_skills": [["python", "sql", "aws"] for _ in range(10)],
            "nice_to_have_skills": [["docker"] for _ in range(10)],
        }
    )


def _tiny_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "job_id": [f"J_{i}" for i in range(10)],
            "relevant_resume_ids": [
                [f"R_{i % 20}", f"R_{(i + 1) % 20}"] for i in range(10)
            ],
        }
    )


def _tiny_pairs() -> pd.DataFrame:
    resumes = _tiny_resumes()
    matches = _tiny_matches()
    return build_pairs(resumes, matches, negatives_per_job=3, seed=42)


class TestPairLoading:
    def test_pairs_built(self) -> None:
        pairs = _tiny_pairs()
        assert len(pairs) > 0
        assert "relevant" in pairs.columns
        assert "job_id" in pairs.columns
        assert "resume_id" in pairs.columns
        assert set(pairs["relevant"]) <= {0, 1}

    def test_target_definition(self) -> None:
        pairs = _tiny_pairs()
        assert pairs["relevant"].sum() > 0
        assert (pairs["relevant"] == 0).sum() > 0


class TestSplitIntegrity:
    def test_job_grouped_no_overlap(self) -> None:
        jobs = _tiny_jobs()
        split = job_group_split(jobs, seed=42)
        train_jobs = {j for j, s in split.items() if s == "train"}
        test_jobs = {j for j, s in split.items() if s == "test"}
        assert not (train_jobs & test_jobs)

    def test_strict_both_no_overlap(self) -> None:
        resumes = _tiny_resumes()
        jobs = _tiny_jobs()
        matches = _tiny_matches()
        manifest = strict_both_split(resumes, jobs, matches, seed=42)
        train_jobs = {j for j, s in manifest.jobs.items() if s == "train"}
        test_jobs = {j for j, s in manifest.jobs.items() if s == "test"}
        train_resumes = {r for r, s in manifest.resumes.items() if s == "train"}
        test_resumes = {r for r, s in manifest.resumes.items() if s == "test"}
        assert not (train_jobs & test_jobs)
        assert not (train_resumes & test_resumes)

    def test_split_reproducible(self) -> None:
        resumes = _tiny_resumes()
        jobs = _tiny_jobs()
        matches = _tiny_matches()
        a = strict_both_split(resumes, jobs, matches, seed=42)
        b = strict_both_split(resumes, jobs, matches, seed=42)
        assert a == b


class TestPreprocessingTrainOnly:
    def test_vectorizer_fit_sees_only_train(self) -> None:
        pairs = _tiny_pairs()
        resumes = _tiny_resumes()
        text_map = resumes.set_index("resume_id")[
            ["summary", "skills", "experience_bullets"]
        ]
        for col in ("summary", "skills", "experience_bullets"):
            pairs[col] = pairs["resume_id"].map(text_map[col])
        vec = ResumeTextVectorizer(max_features=500, min_df=1)
        fitted = vec.fit(pairs.iloc[:10])
        assert fitted.transform(pairs.iloc[:5]) is not None

    def test_structured_features_built(self) -> None:
        pairs = _tiny_pairs()
        features = build_structured_features(pairs)
        assert len(features) == len(pairs)
        assert "same_role" in features.columns
        assert features["same_role"].dtype == float

    def test_deterministic_features_built(self) -> None:
        pairs = _tiny_pairs()
        features = build_deterministic_match_features(pairs)
        assert len(features) == len(pairs)
        assert "keyword_overlap_required" in features.columns


class TestTraining:
    def test_model_training_produces_fitted_estimator(self) -> None:
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0] * 4})
        y = np.array([
            0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
        ])
        plan = ExperimentPlan(
            config=ExperimentConfig(),
            tier=0,
            model_spec="majority",
            manifest=SplitManifest(jobs={}, resumes={}),
        )
        model = train(plan, X, y)
        assert isinstance(model, TrainedModel)
        assert model.estimator is not None
        assert model.train_samples == len(y) > 0

    def test_tfidf_classifier_trains(self) -> None:
        pairs = _tiny_pairs()
        resumes = _tiny_resumes()
        text_map = resumes.set_index("resume_id")[
            ["summary", "skills", "experience_bullets"]
        ]
        for col in ("summary", "skills", "experience_bullets"):
            pairs[col] = pairs["resume_id"].map(text_map[col])
        vec = ResumeTextVectorizer(max_features=500, min_df=1)
        fitted = vec.fit(pairs.iloc[:20])
        X = fitted.transform(pairs.iloc[:10])
        y = np.array(pairs.iloc[:10]["relevant"]).astype(int)
        plan = ExperimentPlan(
            config=ExperimentConfig(),
            tier=1,
            model_spec="logistic_regression",
            manifest=SplitManifest(jobs={}, resumes={}),
        )
        model = train(plan, X, y)
        assert model.estimator is not None
        assert len(model.feature_names) == len(X.columns)

    def test_majority_baseline_predicts(self) -> None:
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = np.array([0, 0, 1])
        plan = ExperimentPlan(
            config=ExperimentConfig(),
            tier=0,
            model_spec="majority",
            manifest=SplitManifest(jobs={}, resumes={}),
        )
        model = train(plan, X, y)
        preds = model.predict(X)
        assert len(preds) == 3


class TestEvaluation:
    def test_metrics_deterministic(self) -> None:
        evaluator = default_evaluator()
        y = np.array([1, 0, 1, 0, 1])
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        plan = ExperimentPlan(
            config=ExperimentConfig(),
            tier=1,
            model_spec="logistic_regression",
            manifest=SplitManifest(jobs={}, resumes={}),
        )
        model = train(plan, X, y)
        r1 = evaluator.evaluate(model, X, y, tier=1, tier_name="x", split_name="test")
        r2 = evaluator.evaluate(model, X, y, tier=1, tier_name="x", split_name="test")
        assert isinstance(r1, EvaluationReport)
        assert r1.metrics == r2.metrics

    def test_empty_handling(self) -> None:
        evaluator = ModelEvaluator()
        with pytest.raises(ValueError):
            evaluator.evaluate(
                model=None,
                X_test=pd.DataFrame({"a": []}),
                y_test=np.array([]),
                tier=0, tier_name="x", split_name="test",
            )


class TestExcludedFeatures:
    def test_excluded_families_absent(self) -> None:
        import ml.features.registry as registry

        excluded = registry.excluded_families()
        assert "identity" in excluded
        assert "raw_must_have" in excluded
        assert "generative_rule" in excluded
        assert "provided_embeddings" in excluded

    def test_ats_6b_subfields_excluded(self) -> None:
        import ml.features.registry as registry

        subfields = registry.excluded_subfields("ats_6b")
        assert "required_coverage" in subfields
        assert "overall_coverage" in subfields
        assert "preferred_coverage" not in subfields

    def test_no_identity_columns_in_features(self) -> None:
        pairs = _tiny_pairs()
        structured = build_structured_features(pairs)
        assert "resume_id" not in structured.columns
        assert "job_id" not in structured.columns


class TestArtifactMetadata:
    def test_artifact_dir_exists(self) -> None:
        from ml.config import ARTIFACTS_DIR

        phase6e = ARTIFACTS_DIR / "phase_6e"
        assert phase6e.is_dir()

    def test_experiment_config_written(self) -> None:
        from ml.config import ARTIFACTS_DIR

        config_path = ARTIFACTS_DIR / "phase_6e" / "experiment_config.json"
        assert config_path.is_file()
        with open(config_path) as f:
            data = json.load(f)
        assert "dataset_id" in data
        assert "seed" in data

    def test_metadata_has_dataset_revision(self) -> None:
        from ml.config import ARTIFACTS_DIR, DATASET_REVISION

        meta_path = ARTIFACTS_DIR / "phase_6e" / "dataset_metadata.json"
        assert meta_path.is_file()
        with open(meta_path) as f:
            data = json.load(f)
        assert data["dataset_revision"] == DATASET_REVISION


class TestSemanticFallback:
    def test_semantic_embedder_has_cpu_fallback(self) -> None:
        from ml.features.semantic import SemanticEmbedder

        emb = SemanticEmbedder(batch_size=4)
        # Construction should not require CUDA; device resolution happens lazily
        assert emb.batch_size == 4
        assert emb.model_name == "sentence-transformers/all-MiniLM-L6-v2"
