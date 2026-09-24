"""Central configuration for the dataset audit workspace."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

DATASET_ID = "michaelozon/candidate-matching-synthetic"
DATASET_REVISION = "178ab864dcad9910c5670d43e4bdbbb901a11f18"

ML_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ML_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

# Local HF hub cache layout:
#   <cache>/hub/datasets--michaelozon--candidate-matching-synthetic/snapshots/<rev>/
HF_CACHE_ROOT = Path(
    os.environ.get(
        "HF_HOME",
        str(Path.home() / ".cache" / "huggingface"),
    )
)
REPO_TYPE = "dataset"


class AuditSettings(BaseModel):
    """Tunable thresholds for the dataset audit."""

    max_top_skills: int = Field(20, description="bars in the skills plots")
    min_phrase_for_duplicate: int = Field(
        20, description="min shared words before a bullet counts as boilerplate"
    )
    duplicate_similarity_words: int = Field(5, description="shared-token threshold")
