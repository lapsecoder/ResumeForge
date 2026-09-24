"""Offline tests for leakage-safe split logic (ml.splits).

Verifies group integrity (all labels of a job stay together), disjointness of
the strict both-group split, deterministic repetition, and coverage of the
strata requested by the design.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml.data.pairs import positive_map
from ml.splits import (
    job_group_split,
    resume_group_split,
    strict_both_split,
)


def _jobs(n: int = 120) -> pd.DataFrame:
    rows = []
    for i in range(n):
        role = "Backend Engineer" if i % 2 == 0 else "Data Scientist"
        seniority = "Mid" if i % 3 == 0 else "Senior"
        rows.append((f"J_{i}", role, seniority, "Tech"))
    return pd.DataFrame(
        rows, columns=["job_id", "job_title", "seniority", "industry"]
    )


def _resumes(n: int = 120) -> pd.DataFrame:
    rows = []
    for i in range(n):
        role = "Backend Engineer" if i % 2 == 0 else "Data Scientist"
        seniority = "Mid" if i % 3 == 0 else "Senior"
        rows.append((f"R_{i}", role, seniority, 4.0, "Tech"))
    return pd.DataFrame(
        rows, columns=["resume_id", "role", "seniority", "years_experience", "industry"]
    )


def _matches(resumes: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "job_id": jobs["job_id"].tolist(),
            "relevant_resume_ids": [
                [f"R_{i % 40}"] for i in range(len(jobs))
            ],
        }
    )


class TestJobGroupSplit:
    def test_all_labels_share_the_job_partition(self) -> None:
        jobs = _jobs()
        split = job_group_split(jobs, seed=42)
        job_ids = jobs["job_id"].tolist()
        assert set(split) == set(job_ids)
        assert set(split.values()) <= {"train", "val", "test"}

    def test_every_partition_nonempty(self) -> None:
        split = job_group_split(_jobs(), seed=42)
        counts = pd.Series(split.values()).value_counts().to_dict()
        assert counts["train"] > counts["test"] > 0
        assert counts["val"] > 0

    def test_deterministic(self) -> None:
        a = job_group_split(_jobs(), seed=42)
        b = job_group_split(_jobs(), seed=42)
        assert a == b

    def test_seed_changes_assignment(self) -> None:
        a = job_group_split(_jobs(), seed=1)
        b = job_group_split(_jobs(), seed=2)
        assert a != b

    def test_stratum_distribution_preserved(self) -> None:
        jobs = _jobs()
        split = job_group_split(jobs, seed=42)
        for (_role, _seniority), group in jobs.groupby(
            ["job_title", "seniority"]
        ):
            members = [split[jid] for jid in group["job_id"]]
            assert "train" in members
            assert "test" in members


class TestResumeGroupSplit:
    def test_disjoint_partitions(self) -> None:
        resumes = _resumes()
        split = resume_group_split(resumes, seed=7)
        parts = {split[rid] for rid in resumes["resume_id"]}
        assert parts == {"train", "val", "test"}
        assert len(split) == len(resumes)

    def test_deterministic(self) -> None:
        a = resume_group_split(_resumes(), seed=7)
        b = resume_group_split(_resumes(), seed=7)
        assert a == b


class TestStrictBothSplit:
    def test_held_out_job_positives_never_in_train(self) -> None:
        resumes = _resumes()
        jobs = _jobs()
        matches = _matches(resumes, jobs)
        manifest = strict_both_split(resumes, jobs, matches, seed=42)
        relevant = positive_map(matches)

        test_positives = {
            rid
            for jid, part in manifest.jobs.items()
            if part == "test"
            for rid in relevant[jid]
        }
        for rid in test_positives:
            assert manifest.resumes[rid] == "test", (
                f"resume {rid} is positive for a test job but sits in "
                f"{manifest.resumes[rid]}"
            )

    def test_partitions_exhaustive_and_disjoint(self) -> None:
        resumes = _resumes()
        jobs = _jobs()
        matches = _matches(resumes, jobs)
        manifest = strict_both_split(resumes, jobs, matches, seed=42)
        assert set(manifest.resumes) == set(resumes["resume_id"])
        assert set(manifest.jobs) == set(jobs["job_id"])

    def test_deterministic(self) -> None:
        resumes = _resumes()
        jobs = _jobs()
        matches = _matches(resumes, jobs)
        a = strict_both_split(resumes, jobs, matches, seed=42)
        b = strict_both_split(resumes, jobs, matches, seed=42)
        assert a == b

    def test_manifest_summary_shapes(self) -> None:
        resumes = _resumes()
        jobs = _jobs()
        matches = _matches(resumes, jobs)
        summary = strict_both_split(resumes, jobs, matches, seed=42).summary()
        assert set(summary["jobs"]) <= {"train", "val", "test"}
        assert set(summary["resumes"]) <= {"train", "val", "test"}
        assert summary["jobs"]["train"] > 0


def test_sizes_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        job_group_split(_jobs(), seed=1, sizes=(0.7, 0.1, 0.1))
