"""Deterministic, leakage-safe splits for the candidate-matching experiments.

The repository ships no partition, so Track C defines three:

* ``job_grouped``  - jobs split 70/10/20 stratified by (job_title, seniority).
  Every one of a job's 30 labels stays together, so the test set contains
  entirely unseen *jobs* while sharing the resume vocabulary.
* ``strict_both``  - also hides the positive resumes of held-out jobs from
  training entirely (job AND resume disjointness). Smaller and noisier, but
  the honest cold-start (new job, new people) estimate.
* ``resume_grouped`` - diagnostic memorisation check: hold out a chunk of
  resumes and see how far a model trained on the rest learns transitive
  relevance through job membership.

All splitting is seeded, column-agnostic to any model, and never touches the
raw matches labels.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ml.data.pairs import positive_map


def _subset_sizes(total: int, sizes: Sequence[float]) -> list[int]:
    if len(sizes) != 3:
        raise ValueError("exactly three splits (train/val/test) are supported")
    if abs(sum(sizes) - 1.0) > 1e-9:
        raise ValueError("split sizes must sum to 1.0")
    counts = [max(0, int(total * size)) for size in sizes]
    remainder = total - sum(counts)
    for index in range(3):
        take = min(remainder, 1)
        counts[index] += take
        remainder -= take
    return counts


def _strata_index(
    frame: pd.DataFrame, key_column: str, stratum_columns: Sequence[str]
) -> dict[str, str]:
    """One pass: ``key -> "|".join(stratum values)`` (or single group)."""
    if not stratum_columns:
        return {}
    index: dict[str, str] = {}
    for _, row in frame.iterrows():
        key = str(row[key_column])
        index[key] = "|".join(str(row[column]) for column in stratum_columns)
    return index


def _stratified_assign(
    keys: Sequence[str],
    strata: dict[str, str],
    *,
    seed: int,
    sizes: Sequence[float],
) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for key in keys:
        grouped.setdefault(strata.get(key, "_unstratified"), []).append(key)

    assignment: dict[str, str] = {}
    rng = random.Random(seed)
    for members in grouped.values():
        shuffled = list(members)
        rng.shuffle(shuffled)
        counts = _subset_sizes(len(shuffled), sizes)
        splits = ["train"] * counts[0] + ["val"] * counts[1] + ["test"] * counts[2]
        for key, split in zip(shuffled, splits, strict=True):
            assignment[key] = split
    return assignment


def _present_columns(frame: pd.DataFrame, preferred: Sequence[str]) -> list[str]:
    return [column for column in preferred if column in frame.columns]


def job_group_split(
    jobs: pd.DataFrame,
    *,
    seed: int,
    sizes: Sequence[float] = (0.7, 0.1, 0.2),
) -> dict[str, str]:
    """Map ``job_id -> split`` with each job kept wholly in one partition."""
    strata_columns = _present_columns(jobs, ("job_title", "seniority"))
    job_ids = sorted(str(value) for value in jobs["job_id"].tolist())
    strata = _strata_index(jobs, "job_id", strata_columns)
    return _stratified_assign(job_ids, strata, seed=seed, sizes=sizes)


def resume_group_split(
    resumes: pd.DataFrame,
    *,
    seed: int,
    sizes: Sequence[float] = (0.7, 0.1, 0.2),
) -> dict[str, str]:
    """Map ``resume_id -> split`` stratified by (role, seniority) when present."""
    strata_columns = _present_columns(resumes, ("role", "seniority"))
    resume_ids = sorted(str(value) for value in resumes["resume_id"].tolist())
    strata = _strata_index(resumes, "resume_id", strata_columns)
    return _stratified_assign(resume_ids, strata, seed=seed, sizes=sizes)


@dataclass(frozen=True)
class SplitManifest:
    """Job and resume partition assignments for one experiment."""

    jobs: dict[str, str]
    resumes: dict[str, str]

    def job_split(self, job_id: str) -> str:
        return self.jobs[job_id]

    def resume_split(self, resume_id: str) -> str:
        return self.resumes.get(resume_id, "train")

    def summary(self) -> dict[str, Any]:
        return {
            "jobs": dict(Counter(self.jobs.values())),
            "resumes": dict(Counter(self.resumes.values())),
        }


def strict_both_split(
    resumes: pd.DataFrame,
    jobs: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    seed: int,
    sizes: Sequence[float] = (0.6, 0.2, 0.2),
) -> SplitManifest:
    """Job-AND-resume disjointness: held-out jobs' positives never train.

    Partition rule: a resume is placed in ``test`` if it is relevant to any
    test job, in ``val`` if it is relevant to any val job (but no test job),
    and ``train`` otherwise. When a resume is relevant to jobs in several
    partitions the most-testward partition wins - this drops a (small)
    number of train labels, which the caller accepts.
    """
    job_split = job_group_split(jobs, seed=seed, sizes=sizes)
    relevant = positive_map(matches)

    test_positives: set[str] = set()
    val_positives: set[str] = set()
    for job_id, split in job_split.items():
        if split == "test":
            test_positives |= relevant.get(job_id, frozenset())
        elif split == "val":
            val_positives |= relevant.get(job_id, frozenset())

    resume_ids = sorted(str(value) for value in resumes["resume_id"].tolist())
    resume_split: dict[str, str] = {}
    for resume_id in resume_ids:
        if resume_id in test_positives:
            resume_split[resume_id] = "test"
        elif resume_id in val_positives:
            resume_split[resume_id] = "val"
        else:
            resume_split[resume_id] = "train"

    return SplitManifest(jobs=job_split, resumes=resume_split)
