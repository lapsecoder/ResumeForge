"""Local semantic embedding layer for resume <-> job matching (Phase 5B).

Transient only: nothing computed here is persisted, and no external service
is contacted. Embeddings come from a bundled ONNX sentence-encoder model that
ships inside this package.
"""

from app.semantic_matching.schemas import SemanticMatchResult
from app.semantic_matching.service import compute_semantic_match

__all__ = ["SemanticMatchResult", "compute_semantic_match"]
