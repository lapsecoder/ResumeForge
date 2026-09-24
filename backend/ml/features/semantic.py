"""MiniLM semantic embedding features.

Generates local embeddings with sentence-transformers/all-MiniLM-L6-v2
and computes cosine similarity features for resume/job pairs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from ml.preprocessing.text import normalize

logger = logging.getLogger(__name__)

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384


def _safe_val(row: pd.Series, col: str) -> object:
    try:
        val = row[col]
    except KeyError:
        return None
    try:
        if pd.isna(val):
            return None
    except (ValueError, TypeError):
        pass
    return val


def _concat_resume_text(row: pd.Series) -> str:
    parts: list[str] = []
    for col in ("summary", "experience_bullets", "skills"):
        val = _safe_val(row, col)
        if val is not None:
            if isinstance(val, (list, tuple)):
                parts.append(" ".join(str(v) for v in val))
            else:
                parts.append(str(val))
    return " ".join(parts)


def _concat_job_text(row: pd.Series) -> str:
    parts: list[str] = []
    for col in ("description", "responsibilities", "requirements"):
        val = _safe_val(row, col)
        if val is not None:
            if isinstance(val, (list, tuple)):
                parts.append(" ".join(str(v) for v in val))
            else:
                parts.append(str(val))
    return " ".join(parts)


def _normalize(text: str) -> str:
    return " ".join(normalize(text).lower().split())


@dataclass(frozen=True)
class SemanticFeatureSet:
    """Precomputed semantic features for all pairs in a split."""

    cosine: pd.Series
    resume_emb: np.ndarray
    job_emb: np.ndarray
    device: str
    model_name: str
    embedding_dim: int
    batch_size: int
    n_pairs: int


class SemanticEmbedder:
    """Generates MiniLM embeddings for resume/job pairs."""

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._device_override = device
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            if self._device_override:
                device = self._device_override
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            logger.info("Loading MiniLM on device=%s", device)
            self._model = SentenceTransformer(self.model_name, device=device)
            return self._model
        except Exception:
            logger.warning("Failed to load MiniLM; falling back to CPU", exc_info=True)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
            return self._model

    @property
    def device(self) -> str:
        model = self._get_model()
        return str(getattr(model, "device", "cpu"))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        from torch import Tensor

        model = self._get_model()
        encoded: Any = model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=False,
            convert_to_numpy=True,
        )
        if isinstance(encoded, Tensor):
            return cast(np.ndarray, encoded.numpy())
        return cast(np.ndarray, np.asarray(encoded, dtype=np.float32))

    def embed_pairs(self, pairs: pd.DataFrame) -> SemanticFeatureSet:
        resume_texts = [
            _normalize(_concat_resume_text(row)) for _, row in pairs.iterrows()
        ]
        job_texts = [
            _normalize(_concat_job_text(row)) for _, row in pairs.iterrows()
        ]

        logger.info("Embedding %d resume texts", len(resume_texts))
        resume_embs = self.embed_texts(resume_texts)
        logger.info("Embedding %d job texts", len(job_texts))
        job_embs = self.embed_texts(job_texts)

        cosines = _batch_cosine(resume_embs, job_embs)

        return SemanticFeatureSet(
            cosine=pd.Series(cosines, index=pairs.index, dtype=float),
            resume_emb=resume_embs,
            job_emb=job_embs,
            device=self.device,
            model_name=self.model_name,
            embedding_dim=_EMBEDDING_DIM,
            batch_size=self.batch_size,
            n_pairs=len(pairs),
        )


def _batch_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.sum(a_norm * b_norm, axis=1)  # type: ignore[no-any-return]


def build_semantic_features(
    pairs: pd.DataFrame, emb: SemanticFeatureSet
) -> pd.DataFrame:
    """Build a DataFrame of semantic features from precomputed embeddings."""
    return pd.DataFrame(
        {"minilm_cosine": emb.cosine.values},
        index=pairs.index,
        dtype=float,
    )
