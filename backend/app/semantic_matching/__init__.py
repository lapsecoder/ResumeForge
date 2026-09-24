"""Local semantic embedding layer for resume <-> job matching (Phase 5B).

Transient only: nothing computed here is persisted, and no external service
is contacted. Embeddings come from a local sentence-encoder model whose
weights live in the machine's Hugging Face cache.
"""

from app.semantic_matching.schemas import SemanticMatchResult
from app.semantic_matching.service import compute_semantic_match

__all__ = ["SemanticMatchResult", "compute_semantic_match"]
