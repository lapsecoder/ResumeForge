"""Feature-transformer contracts for Track C (implemented in Phase 6E).

The registry (``ml/features/registry.py``) decides *which* families are
allowed; this module sets out the shape each transformer must obey:

* stateless for the deterministic families (same inputs -> same columns);
* ``FIT_ON_TRAIN`` only for families whose representation is learned
  (TF-IDF vocabulary, scalers, encoders, any dimensional projection); the
  fit step sees only the training pairs.
* returned frames are plain feature tables with no job/resume identity.

Phase 6D ships no transformers and imports no sklearn. The classes below are
interfaces against which 6E is written.
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

# Families whose representation MUST be fit on the training split only.
FIT_ON_TRAIN: tuple[str, ...] = (
    "resume_text",
    "job_text",
    "structured_resume",
    "structured_job",
)

# The ablation canary: families with quasi-generative leakage. A final report
# must state macro-F1 with and without this group to separate genuine matching
# signal from rule reproduction.
CANARY_FAMILIES: tuple[str, ...] = ("deterministic_match", "ats_6b")


class StatelessTransformer(Protocol):
    """Deterministic feature construction with no learned state."""

    def transform(self, pairs: pd.DataFrame) -> pd.DataFrame: ...


class TrainableTransformer(Protocol):
    """Transformer whose representation is learned from training pairs only."""

    def fit(self, pairs_train: pd.DataFrame, y_train: Any) -> "FittedTransformer": ...


class FittedTransformer(Protocol):
    """Trained transformer applicable to any split's pairs."""

    def transform(self, pairs: pd.DataFrame) -> pd.DataFrame: ...
