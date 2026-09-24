"""Offline tests for the pair-target definition (Track C data contract).

Verifies that positives are preserved verbatim, negatives are sampled
deterministically per job, and the label framing is the one documented in the
6D design (sampled negatives are "not published as relevant").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.data.pairs import (
    NEGATIVE,
    POSITIVE,
    TARGET_COLUMN,
    build_pairs,
    label_counts,
    positive_map,
)


def _fixtures() -> tuple[pd.DataFrame, pd.DataFrame]:
    resumes = pd.DataFrame(
        {
            "resume_id": [f"R_{i}" for i in range(5)],
            "role": [f"role_{i}" for i in range(5)],
        }
    )
    matches = pd.DataFrame(
        {
            "job_id": ["J_1", "J_2"],
            "relevant_resume_ids": [["R_0", "R_1"], ["R_0"]],
        }
    )
    return resumes, matches


class TestPositiveMap:
    def test_string_and_numpy_values(self) -> None:
        matches = pd.DataFrame(
            {
                "job_id": ["J_1"],
                "relevant_resume_ids": [np.array(["R_0", "R_1"])],
            }
        )
        assert positive_map(matches) == {
            "J_1": frozenset({"R_0", "R_1"})
        }

    def test_none_handled(self) -> None:
        matches = pd.DataFrame(
            {"job_id": ["J_1"], "relevant_resume_ids": [None]}
        )
        assert positive_map(matches) == {"J_1": frozenset()}


class TestBuildPairs:
    def test_positives_preserved_and_negatives_balanced(self) -> None:
        resumes, matches = _fixtures()
        pairs = build_pairs(resumes, matches, negatives_per_job=2, seed=7)
        counts = label_counts(pairs)
        assert counts["positive"] == 3
        assert counts["negative"] == 4

    def test_per_job_negative_cap(self) -> None:
        resumes, matches = _fixtures()
        pairs = build_pairs(resumes, matches, negatives_per_job=99, seed=7)
        counts = label_counts(pairs)
        assert counts["positive"] == 3
        assert counts["negative"] == 3 + 4  # J_1 skips R_0,R_1; J_2 skips R_0

    def test_deterministic_across_runs(self) -> None:
        resumes, matches = _fixtures()
        first = build_pairs(resumes, matches, negatives_per_job=2, seed=7)
        second = build_pairs(resumes, matches, negatives_per_job=2, seed=7)
        pd.testing.assert_frame_equal(first, second)

    def test_seed_changes_negatives(self) -> None:
        resumes, matches = _fixtures()
        a = build_pairs(resumes, matches, negatives_per_job=2, seed=1)
        b = build_pairs(resumes, matches, negatives_per_job=2, seed=2)
        assert not a.equals(b)

    def test_never_samples_a_published_positive(self) -> None:
        resumes, matches = _fixtures()
        pairs = build_pairs(resumes, matches, negatives_per_job=10, seed=3)
        j1_negatives = set(
            pairs.loc[
                (pairs["job_id"] == "J_1") & (pairs[TARGET_COLUMN] == NEGATIVE),
                "resume_id",
            ]
        )
        assert j1_negatives <= {"R_2", "R_3", "R_4"}

    def test_resume_allowlist_restricts_pool(self) -> None:
        resumes, matches = _fixtures()
        pairs = build_pairs(
            resumes, matches, negatives_per_job=10, seed=3,
            resume_allowlist=["R_0", "R_1", "R_2", "R_3", "R_4"],
        )
        assert set(pairs["resume_id"]) <= {"R_0", "R_1", "R_2", "R_3", "R_4"}

    def test_negative_may_be_positive_for_other_job(self) -> None:
        resumes, matches = _fixtures()
        pairs = build_pairs(resumes, matches, negatives_per_job=5, seed=5)
        # R_0 is positive for both jobs; sampling must never flag it negative
        # for a job that marks it relevant.
        for _, row in pairs.iterrows():
            if row["job_id"] == "J_1" and row["resume_id"] == "R_0":
                assert row[TARGET_COLUMN] == POSITIVE
            if row["job_id"] == "J_2" and row["resume_id"] == "R_0":
                assert row[TARGET_COLUMN] == POSITIVE

    def test_negative_validation(self) -> None:
        resumes, matches = _fixtures()
        with pytest.raises(ValueError):
            build_pairs(resumes, matches, negatives_per_job=-1, seed=1)


class TestLabelCounts:
    def test_counts(self) -> None:
        pairs = pd.DataFrame({TARGET_COLUMN: [1, 1, 0]})
        assert label_counts(pairs) == {
            "positive": 2, "negative": 1, "rows": 3
        }
