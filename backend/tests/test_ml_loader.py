"""Offline tests for the ML data loader and value flattening.

Fully offline: the hub-cache layout is reconstructed inside a tmp dir and
``ml.loader.HF_CACHE_ROOT`` is pointed at it. No network calls happen here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ml.loader as loader
from ml.audit._helpers import flatten_value
from ml.loader import DatasetUnavailableError


def _frames():
    resumes = pd.DataFrame(
        {
            "resume_id": ["R_1", "R_2"],
            "role": ["Backend Engineer", "Data Scientist"],
            "seniority": ["Mid", "Senior"],
            "years_experience": [4, 8],
            "industry": ["Tech", "Finance"],
            "education": ["B.Tech", "M.Sc"],
            "skills": [["Python", "Docker"], ["Python"]],
            "summary": ["s1", "s2"],
            "experience_bullets": [["b1", "b2"], ["b3"]],
        }
    )
    jobs = pd.DataFrame(
        {
            "job_id": ["J_1"],
            "job_title": ["Backend Engineer"],
            "seniority": ["Mid"],
            "industry": ["Tech"],
            "must_have_skills": [["Python", "Docker"]],
            "nice_to_have_skills": [[]],
            "description": ["desc"],
            "responsibilities": [["build things"]],
            "requirements": [[]],
        }
    )
    matches = pd.DataFrame(
        {
            "job_id": ["J_1"],
            "relevant_resume_ids": [["R_1", "R_2"]],
        }
    )
    return resumes, jobs, matches


@pytest.fixture()
def fake_cache(tmp_path, monkeypatch):
    root = tmp_path / "hub"
    snapshot = (
        root
        / "datasets--michaelozon--candidate-matching-synthetic"
        / "snapshots"
        / "178ab864dcad9910c5670d43e4bdbbb901a11f18"
    )
    resumes, jobs, matches = _frames()
    (snapshot / "resumes").mkdir(parents=True, exist_ok=True)
    (snapshot / "jobs").mkdir(parents=True, exist_ok=True)
    (snapshot / "matches").mkdir(parents=True, exist_ok=True)
    resumes.to_parquet(snapshot / "resumes" / "train-00000-of-00001.parquet")
    jobs.to_parquet(snapshot / "jobs" / "train-00000-of-00001.parquet")
    matches.to_parquet(snapshot / "matches" / "train-00000-of-00001.parquet")
    monkeypatch.setattr(loader, "HF_CACHE_ROOT", tmp_path)
    return tmp_path


class TestFlattenValue:
    def test_plain_list(self) -> None:
        assert flatten_value(["a", "b"]) == ["a", "b"]

    def test_numpy_array(self) -> None:
        value = np.array(["R_1", "R_2", "R_3"])
        assert flatten_value(value) == ["R_1", "R_2", "R_3"]

    def test_nested_numpy_array(self) -> None:
        value = np.array([["R_1", "R_2"]])
        assert flatten_value(value) == ["R_1", "R_2"]

    def test_none_and_scalar(self) -> None:
        assert flatten_value(None) == []
        assert flatten_value("x") == ["x"]
        assert flatten_value(5) == [5]

    def test_tuple(self) -> None:
        assert flatten_value(("a",)) == ["a"]


class TestLoaderResolution:
    def test_resolves_local_snapshot(self, fake_cache) -> None:
        root = loader._local_snapshot_root()
        assert root is not None
        assert root.name == "178ab864dcad9910c5670d43e4bdbbb901a11f18"

    def test_load_bundle_offline(self, fake_cache) -> None:
        bundle = loader.load_bundle(allow_download=False)
        assert bundle.frame_counts == {"resumes": 2, "jobs": 1, "matches": 1}
        assert list(bundle.resumes["resume_id"]) == ["R_1", "R_2"]
        assert bundle.matches["relevant_resume_ids"].map(flatten_value).tolist() == [
            ["R_1", "R_2"]
        ]

    def test_missing_cache_raises_without_download(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(loader, "HF_CACHE_ROOT", tmp_path)
        with pytest.raises(DatasetUnavailableError):
            loader.load_bundle(allow_download=False)

    def test_frame_counts_are_ints(self, fake_cache) -> None:
        bundle = loader.load_bundle(allow_download=False)
        assert isinstance(bundle.frame_counts["resumes"], int)
