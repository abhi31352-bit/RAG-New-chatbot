"""The single place a SentenceTransformer is ever constructed.

architecture.md D1: the embedding model is fixed
(`sentence-transformers/all-MiniLM-L6-v2`, 384-dim) and the SAME model must
embed both corpus chunks and user questions.

Why this module is so defensive: two model instances (or one instance plus a
different revision) silently produce vectors from different spaces. Nothing
crashes, retrieval just quietly returns garbage, and the failure surfaces as
"the chatbot gives confident wrong answers" rather than as an error. Phase 8
locks this down with tests/test_embedding_consistency.py.

Usage:
    embedder = get_embedder()
    vectors  = embedder.encode(["some text"])
"""
from __future__ import annotations

import threading
from typing import List, Optional, Sequence

import numpy as np

from .config import get_config


class EmbeddingDimensionError(RuntimeError):
    """Raised when the model does not produce EMBED_DIM vectors."""


class Embedder:
    """Lazily-loaded, shared sentence-transformer wrapper."""

    def __init__(self, model_name: Optional[str] = None, expected_dim: Optional[int] = None):
        config = get_config()
        self.model_name = model_name or config.embed_model
        self.expected_dim = expected_dim or config.embed_dim
        self._model = None
        self._lock = threading.Lock()

    # -- model loading ---------------------------------------------------

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name)
                    self._verify_model()
        return self._model

    def _verify_model(self) -> None:
        """Fail loudly now rather than silently retrieving garbage later."""
        dim = self._model.get_sentence_embedding_dimension()
        if dim != self.expected_dim:
            raise EmbeddingDimensionError(
                f"Model {self.model_name!r} produces {dim}-dim vectors but "
                f"EMBED_DIM is {self.expected_dim}. A corpus built with one "
                f"model cannot be queried with another -- rebuild the index."
            )

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def identity(self) -> str:
        """Stable fingerprint of the loaded model, for the consistency test."""
        return f"{self.model_name}@{self.dimension}d"

    # -- encoding --------------------------------------------------------

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Embed texts to a (n, dim) float32 array, L2-normalised.

        Normalisation makes cosine similarity a plain dot product, which is
        what makes the Chroma `cosine` space the correct choice for this model.
        """
        items = list(texts)
        if not items:
            return np.zeros((0, self.expected_dim), dtype=np.float32)

        vectors = self.model.encode(
            items,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[1] != self.expected_dim:
            raise EmbeddingDimensionError(
                f"Expected (n, {self.expected_dim}) from {self.model_name!r}, "
                f"got shape {vectors.shape}"
            )
        return vectors

    def encode_one(self, text: str) -> List[float]:
        """Embed a single string, returned as a plain list for Chroma."""
        return self.encode([text])[0].tolist()


_EMBEDDER: Optional[Embedder] = None
_EMBEDDER_LOCK = threading.Lock()


def get_embedder() -> Embedder:
    """Process-wide singleton. Callers must share this, never construct one."""
    global _EMBEDDER
    if _EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER is None:
                _EMBEDDER = Embedder()
    return _EMBEDDER


__all__ = ["Embedder", "EmbeddingDimensionError", "get_embedder"]
