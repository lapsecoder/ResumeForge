"""Embedding provider abstractions and the local ONNX Runtime model.

The heavy model stack is imported lazily, so importing this module never
loads it: the API process stays fast and the unit suite never needs the
libraries installed. Tests inject a fake session/tokenizer via
``_get_session_class()`` / ``_get_tokenizer_class()`` and never touch disk
or the network.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Callable, Protocol, cast

import numpy as np
from pydantic import BaseModel

from app.semantic_matching.config import (
    DEFAULT_MODEL_DIMENSION,
    MODEL_DIR,
    MODEL_FILENAME,
    MODEL_LICENSE,
    MODEL_SOURCE_DEFAULT,
    TOKENIZER_FILENAME,
    semantic_settings,
)

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
    source: str = MODEL_SOURCE_DEFAULT
    license: str = MODEL_LICENSE


class EmbeddingProvider(Protocol):
    """Anything that turns text lists into float vectors."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...

    def metadata(self) -> ModelMetadata: ...


def _mean_pool_and_normalize(
    last_hidden: np.ndarray, attention_mask: np.ndarray
) -> np.ndarray:
    """Mean-pool token vectors over the attention mask, then L2-normalize."""
    mask: np.ndarray = attention_mask[:, :, None].astype(np.float32)
    summed: np.ndarray = np.sum(last_hidden * mask, axis=1)
    counts: np.ndarray = np.maximum(np.sum(mask, axis=1), 1e-9)
    pooled: np.ndarray = summed / counts
    norms: np.ndarray = np.linalg.norm(pooled, axis=1, keepdims=True)
    return np.asarray(pooled / np.maximum(norms, 1e-9), dtype=np.float32)  # type: ignore[no-any-return]


class LocalSentenceTransformerProvider:
    """Provider backed by the bundled all-MiniLM-L6-v2 ONNX model on CPU.

    The quantised ONNX checkpoint and matching ``tokenizers`` tokenizer ship
    inside this package, so the model loads locally with no download, GPU, or
    external service. The session and tokenizer load once per process
    (lazily, on first use) and are kept for reuse.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or semantic_settings.model_name
        self._session: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None
        self._dimension: int | None = None

    # -- lifecycle ---------------------------------------------------------

    def _get_session_class(self) -> Callable[..., Any]:
        """Return the ``onnxruntime.InferenceSession`` class, lazily.

        Kept as a separate indirection so tests can substitute a fake session
        class without the real library being installed.
        """
        try:
            import onnxruntime  # noqa: PLC0415

            return cast(Callable[..., Any], onnxruntime.InferenceSession)
        except ImportError as exc:  # pragma: no cover - depends on install state
            raise ModelLoadError(
                "The onnxruntime library is not installed. Install it with "
                "`pip install onnxruntime` to enable local semantic matching."
            ) from exc

    def _get_tokenizer_class(self) -> Callable[..., Any]:
        """Return the ``tokenizers.Tokenizer`` class, lazily.

        Same seam style as ``_get_session_class``: tests inject a fake.
        """
        try:
            import tokenizers  # noqa: PLC0415

            return cast(Callable[..., Any], tokenizers.Tokenizer)
        except ImportError as exc:  # pragma: no cover - depends on install state
            raise ModelLoadError(
                "The tokenizers library is not installed. Install it with "
                "`pip install tokenizers` to enable local semantic matching."
            ) from exc

    def _select_device(self) -> str:
        """ONNX Runtime runs on CPU only in this deployment."""
        return "cpu"

    def _load(self) -> None:
        """Instantiate the underlying session and tokenizer exactly once."""
        if self._session is not None:
            return
        device = self._select_device()
        try:
            session_class = self._get_session_class()
            tokenizer_class = cast(Any, self._get_tokenizer_class())
            model_path = str(MODEL_DIR / MODEL_FILENAME)
            session = session_class(model_path, providers=["CPUExecutionProvider"])
            tokenizer = tokenizer_class.from_file(str(MODEL_DIR / TOKENIZER_FILENAME))
        except (ModelLoadError, SemanticError):
            raise
        except Exception as exc:  # pragma: no cover - varies by environment
            raise ModelLoadError(
                "The bundled local semantic model could not be loaded."
            ) from exc
        self._session = session
        self._tokenizer = tokenizer
        self._device = device
        try:
            dim = int(session.get_outputs()[0].shape[-1])
        except (AttributeError, IndexError, TypeError, ValueError):
            dim = DEFAULT_MODEL_DIMENSION
        self._dimension = dim if dim > 0 else DEFAULT_MODEL_DIMENSION
        logger.info(
            "Semantic model loaded: %s on %s (bundled ONNX)", self._model_name, device
        )

    # -- interface ---------------------------------------------------------

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` into a list of dense float vectors (same order)."""
        self._load()
        if not texts:
            return []
        batch_size = semantic_settings.batch_size
        vectors: list[np.ndarray] = []
        try:
            for start in range(0, len(texts), batch_size):
                chunk = list(texts[start : start + batch_size])
                encoded = self._tokenizer.encode_batch(chunk)
                input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
                attention_mask = np.array(
                    [e.attention_mask for e in encoded], dtype=np.int64
                )
                token_type_ids = np.zeros_like(input_ids)
                last_hidden = self._session.run(
                    None,
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids,
                    },
                )[0]
                vectors.append(_mean_pool_and_normalize(last_hidden, attention_mask))
        except Exception as exc:  # pragma: no cover - varies by environment
            raise InferenceError("Embedding inference failed.") from exc
        try:
            return [list(map(float, row)) for vector in vectors for row in vector]
        except (TypeError, ValueError) as exc:
            raise InferenceError(
                "Embedding inference produced invalid output."
            ) from exc

    def metadata(self) -> ModelMetadata:
        """Report non-sensitive model metadata (no input content)."""
        return ModelMetadata(
            model_name=self._model_name,
            # The bundled ONNX checkpoint is exported from this exact model;
            # the fix version of the checkpoint can be pinned here later.
            model_version=self._model_name,
            model_dimension=self._dimension or DEFAULT_MODEL_DIMENSION,
            device=self._device or "cpu",
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
