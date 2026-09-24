"""Thin wrapper turning text units into vectors via an embedding provider."""

from __future__ import annotations

from app.semantic_matching.model import (
    EmbeddingProvider,
    InferenceError,
    ModelMetadata,
    get_embedding_provider,
)
from app.semantic_matching.text_builder import SemanticUnit


class Embedder:
    """Batch-embeds :class:`SemanticUnit` texts with one provider."""

    def __init__(self, provider: EmbeddingProvider | None = None) -> None:
        self._provider: EmbeddingProvider | None = provider

    def ensure_provider(self) -> EmbeddingProvider:
        if self._provider is None:
            self._provider = get_embedding_provider()
        return self._provider

    def embed_units(self, units: list[SemanticUnit]) -> dict[str, list[float]]:
        """Embed units in order and return ``{unit.name: vector}``."""
        if not units:
            return {}
        provider = self.ensure_provider()
        vectors = provider.embed_texts([unit.text for unit in units])
        if len(vectors) != len(units):
            raise InferenceError("Embedder returned an unexpected number of vectors.")
        return {
            unit.name: vector
            for unit, vector in zip(units, vectors, strict=True)
            if all(isinstance(value, float) for value in vector)
        }

    def metadata(self) -> ModelMetadata:
        return self.ensure_provider().metadata()
