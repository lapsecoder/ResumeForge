"""Embedding provider abstractions and the local sentence-transformer model.

The heavy model library is imported lazily, so importing this module never
loads it: the API process stays fast and the unit suite never needs the
library installed. Tests inject a fake sentence-transformer class via
``_get_st_class()`` and never touch the network.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Callable, Protocol, cast

from pydantic import BaseModel

from app.semantic_matching.config import MODEL_LICENSE, semantic_settings

logger = logging.getLogger(__name__)


class SemanticError(Exception):
    """Base class for semantic-layer failures (never includes user content)."""


class ModelLoadError(SemanticError):
    """The local embedding model could not be loaded."""


class InferenceError(SemanticError):
    """Embedding inference failed or produced malformed output."""


class ModelMetadata(BaseModel):
    """Non-sensitive metadata about the model backing a result."""

    model_name: str
    model_version: str = ""
    model_dimension: int
    device: str
    source: str = "local Hugging Face cache"
    license: str = MODEL_LICENSE


class EmbeddingProvider(Protocol):
    """Anything that turns text lists into float vectors."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...

    def metadata(self) -> ModelMetadata: ...


class LocalSentenceTransformerProvider:
    """Provider backed by :mod:`sentence-transformers` on the local machine.

    The model is loaded once per process (lazily, on first use) and kept in
    memory/VRAM for reuse. Device selection: CUDA when available, else CPU.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or semantic_settings.model_name
        self._model: Any = None
        self._device: str | None = None
        self._dimension: int | None = None

    # -- lifecycle ---------------------------------------------------------

    def _get_st_class(self) -> Callable[..., Any]:
        """Return the ``SentenceTransformer`` class, importing lazily.

        Kept as a separate indirection so tests can substitute a fake class
        without the real library being installed.
        """
        try:
            from sentence_transformers import SentenceTransformer

            return cast(Callable[..., Any], SentenceTransformer)
        except ImportError as exc:  # pragma: no cover - depends on install state
            raise ModelLoadError(
                "The sentence-transformers library is not installed. Install it "
                "with `pip install sentence-transformers` to enable local "
                "semantic matching."
            ) from exc

    def _select_device(self) -> str:
        """Pick 'cuda' when a usable GPU is present, else 'cpu'."""
        try:
            import torch  # noqa: PLC0415

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover - depends on install state
            return "cpu"

    def _load(self) -> None:
        """Instantiate the underlying model exactly once."""
        if self._model is not None:
            return
        device = self._select_device()
        try:
            st_class = self._get_st_class()
            model = st_class(self._model_name, device=device)
            dimension = getattr(model, "get_sentence_embedding_dimension", None)
            self._dimension = (
                int(dimension())
                if callable(dimension)
                else semantic_settings.model_dimension
            )
        except (ModelLoadError, SemanticError):
            raise
        except Exception as exc:  # pragma: no cover - varies by environment
            raise ModelLoadError(
                "The local semantic model could not be loaded."
            ) from exc
        self._model = model
        self._device = device
        logger.info("Semantic model loaded: %s on %s", self._model_name, device)

    # -- interface ---------------------------------------------------------

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` into a list of dense float vectors (same order)."""
        self._load()
        try:
            encoded = self._model.encode(
                list(texts), batch_size=semantic_settings.batch_size
            )
        except Exception as exc:  # pragma: no cover - varies by environment
            raise InferenceError("Embedding inference failed.") from exc
        try:
            return [list(map(float, row)) for row in encoded]
        except (TypeError, ValueError) as exc:
            raise InferenceError(
                "Embedding inference produced invalid output."
            ) from exc

    def metadata(self) -> ModelMetadata:
        """Report non-sensitive model metadata (no input content)."""
        self._load()
        return ModelMetadata(
            model_name=self._model_name,
            # The HF repo id identifies a specific published model; a pinned
            # revision can be added here later without changing the contract.
            model_version=self._model_name,
            model_dimension=self._dimension or semantic_settings.model_dimension,
            device=self._device or "unknown",
        )


_PROVIDER: LocalSentenceTransformerProvider | None = None


def get_embedding_provider() -> LocalSentenceTransformerProvider:
    """Return the process-wide singleton provider (created lazily)."""
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = LocalSentenceTransformerProvider()
    return _PROVIDER


def reset_embedding_provider() -> None:
    """Drop the singleton; used by tests to verify lazy-loading behaviour."""
    global _PROVIDER
    _PROVIDER = None
