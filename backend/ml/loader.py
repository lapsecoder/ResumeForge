"""Pinned, offline-friendly loader for the candidate-matching-synthetic dataset.

Why not ``datasets.load_dataset``:

- the HF ``datasets`` default config only exposes the ``resumes`` split (the
  raw ``jobs`` and ``matches`` parquet files are ignored);
- passing ``data_files=`` with mixed schemas hits a
  ``datasets.table.CastError`` because every file is cast to the first
  file's features.

Instead we read the raw parquet files directly with pandas, resolved through
the pinned snapshot revision. The snapshot is already present in the local HF
hub cache (verified during Phase 6C exploration); if it is absent the loader
falls back to a pinned network download.

Privacy: nothing loaded through this module ever leaves the machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml.config import (
    DATASET_ID,
    DATASET_REVISION,
    HF_CACHE_ROOT,
    REPO_TYPE,
)

_LOGICAL_PATHS = {
    "resumes": (
        "resumes/train-00000-of-00001.parquet",
        "data/resumes-00000-of-00001.parquet",
    ),
    "jobs": ("jobs/train-00000-of-00001.parquet",),
    "matches": ("matches/train-00000-of-00001.parquet",),
}

_SNAPSHOT_SUFFIX = DATASET_ID.replace("/", "--")


class DatasetUnavailableError(RuntimeError):
    """Raised when the dataset cannot be resolved offline and downloads are off."""


@dataclass(frozen=True)
class DatasetBundle:
    """The three raw tables of the synthetic dataset."""

    resumes: pd.DataFrame
    jobs: pd.DataFrame
    matches: pd.DataFrame

    @property
    def frame_counts(self) -> dict[str, int]:
        return {
            "resumes": len(self.resumes),
            "jobs": len(self.jobs),
            "matches": len(self.matches),
        }


def _candidate_snapshot_dirs() -> list[Path]:
    hub = HF_CACHE_ROOT / "hub"
    base = hub / f"datasets--{_SNAPSHOT_SUFFIX}"
    if not base.is_dir():
        return []
    snapshots = base / "snapshots"
    if not snapshots.is_dir():
        return []
    ordered = sorted(
        (p for p in snapshots.iterdir() if (p / "refs").exists() or list(p.iterdir())),
        key=lambda p: p.name,
    )
    return ordered


def _local_snapshot_root() -> Path | None:
    """Snapshot dir matching the pinned revision, else the newest available."""
    for candidate in _candidate_snapshot_dirs():
        if candidate.name == DATASET_REVISION:
            return candidate
    if _candidate_snapshot_dirs():
        return _candidate_snapshot_dirs()[-1]
    return None


def _resolve_local_path(kind: str) -> Path | None:
    root = _local_snapshot_root()
    if root is None:
        return None
    for relative in _LOGICAL_PATHS[kind]:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    for parquet in root.rglob("*.parquet"):
        if kind == "resumes" and parquet.name.startswith("resumes-"):
            return parquet
    return None


def _download_path(kind: str) -> Path:
    """Network fallback (pinned revision). Requires the ``network`` allowance."""
    from huggingface_hub import hf_hub_download

    relative = _LOGICAL_PATHS[kind][0]
    path = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type=REPO_TYPE,
        revision=DATASET_REVISION,
        filename=relative,
    )
    return Path(path)


def load_bundle(*, allow_download: bool = True) -> DatasetBundle:
    """Load resumes/jobs/matches as pandas frames from the pinned snapshot.

    Prefers the local HF hub cache; only hits the network when the snapshot is
    missing and ``allow_download`` is set.
    """
    paths: dict[str, Path] = {}
    for kind in ("resumes", "jobs", "matches"):
        local = _resolve_local_path(kind)
        if local is not None:
            paths[kind] = local
            continue
        if not allow_download:
            raise DatasetUnavailableError(
                f"{kind} not found in local HF cache (revision {DATASET_REVISION}) "
                "and network downloads are disabled"
            )
        paths[kind] = _download_path(kind)

    return DatasetBundle(
        resumes=pd.read_parquet(paths["resumes"]),
        jobs=pd.read_parquet(paths["jobs"]),
        matches=pd.read_parquet(paths["matches"]),
    )
