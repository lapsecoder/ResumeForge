"""Configuration for the local semantic embedding layer.

Settings derive from environment variables with the ``SEMANTIC_`` prefix
(e.g. ``SEMANTIC_MODEL_NAME``). All defaults target a zero-cost, local,
CPU-capable setup.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Model identity reported in metadata. The weights run from the bundled ONNX
#: artifact in ``models/`` below — nothing is downloaded at runtime.
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

#: Bundled ONNX + tokenizer artifacts shipped inside this package.
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_FILENAME = "model_qint8_avx512_vnni.onnx"
TOKENIZER_FILENAME = "tokenizer.json"
VOCAB_FILENAME = "vocab.txt"

#: Where the embeddings actually come from (reported in API metadata).
MODEL_SOURCE_DEFAULT = "bundled in-app model (no runtime download)"

#: Embedding dimension of the default model. Overridden at runtime with the
#: value reported by the loaded model when available.
DEFAULT_MODEL_DIMENSION = 384

#: Documented licence of the default model weights.
MODEL_LICENSE = "Apache-2.0"

#: Semantic implementation version — bumped when the semantic layer changes
#: its public contract or inference behaviour.
SEMANTIC_IMPLEMENTATION_VERSION = "2.0"


class SemanticSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SEMANTIC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #: Local sentence-encoder model (name or HF repo id).
    model_name: str = DEFAULT_MODEL_NAME

    #: Fallback dimension used when the loaded model does not report one.
    model_dimension: int = DEFAULT_MODEL_DIMENSION

    #: Batch size for embedding calls. Trades memory for speed.
    batch_size: int = 32

    # --- Similarity thresholds ---
    # Operate on normalized similarity in [0, 1]; the same settings drive the
    # API response metadata plus the high/moderate/low category buckets.
    high_similarity_threshold: float = 0.65
    moderate_similarity_threshold: float = 0.40

    @field_validator("high_similarity_threshold", "moderate_similarity_threshold")
    @classmethod
    def _threshold_within_bounds(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("similarity thresholds must be within [0, 1]")
        return value

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> "SemanticSettings":
        if self.moderate_similarity_threshold >= self.high_similarity_threshold:
            raise ValueError(
                "moderate_similarity_threshold must be below high_similarity_threshold"
            )
        return self


@lru_cache
def get_semantic_settings() -> SemanticSettings:
    """Return a cached SemanticSettings instance."""
    return SemanticSettings()


semantic_settings = get_semantic_settings()
