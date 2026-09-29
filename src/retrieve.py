"""Retrieval over the persisted ChromaDB index.

Read-only by construction: this module opens the collection and never writes,
never embeds the corpus, and never creates the collection. Ingestion belongs to
`src/ingest.py` (architecture.md ADR-001), which is what makes "ingestion runs
once" structural rather than a convention.

The scheme filter is the load-bearing part, not a tidy-up. Measured in Phase 3,
the five `Key scheme facts` chunks are 0.92-0.98 cosine-similar to each other
because they share an identical field layout and differ only in values. An
unfiltered query therefore cannot pick the right scheme from the embedding
alone -- "What is the exit load?" ranks the ELSS chunk ("Exit load: Nil") first
even though that scheme has no exit load. `detect_scheme` plus the `where`
clause is what makes that question correct instead of confidently wrong.

Usage:
    retriever = Retriever()
    hits = retriever.search("What is the exit load on HDFC Small Cap?")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import get_config
from .sources import detect_scheme

LOGGER = logging.getLogger("retrieve")


@dataclass
class RetrievedChunk:
    """One retrieved chunk, already trimmed to what a citation needs."""

    chunk_id: str
    text: str
    scheme_id: str
    scheme_name: str
    section: Optional[str]
    source_url: str
    last_updated: str
    score: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "scheme_id": self.scheme_id,
            "scheme_name": self.scheme_name,
            "section": self.section,
            "source_url": self.source_url,
            "last_updated": self.last_updated,
            "score": round(self.score, 4),
        }


class Retriever:
    """Loads the collection once; loads the embedding model on first search."""

    def __init__(
        self,
        k_fetch: Optional[int] = None,
        k_context: Optional[int] = None,
        max_per_section: int = 2,
        dedup_threshold: float = 0.97,
    ) -> None:
        """
        Args:
            k_fetch: candidates to pull from the store before trimming.
            k_context: how many chunks reach the model.
            max_per_section: cap on chunks from one section, so a query about
                fees cannot return four rows of the same holdings table.
            dedup_threshold: cosine similarity above which two chunks are
                treated as the same passage.
        """
        self.config = get_config()
        self.k_fetch = k_fetch or self.config.k_fetch
        self.k_context = k_context or self.config.k_context
        self.max_per_section = max_per_section
        self.dedup_threshold = dedup_threshold
        self._collection = None
        self._embedder = None
        # Per-instance, not a class attribute: a class-level dict would be
        # shared by every Retriever and would leak vectors between searches.
        self._embeddings: Dict[str, Optional[np.ndarray]] = {}

    # -- lazy loading ----------------------------------------------------

    @property
    def collection(self):
        if self._collection is None:
            import chromadb

            # Check the store exists BEFORE constructing the client:
            # chromadb.PersistentClient() initialises the directory and writes a
            # chroma.sqlite3 on open, so probing via the client first would
            # create an empty store as a side effect of a read-only module.
            if not (self.config.chroma_dir / "chroma.sqlite3").exists():
                raise FileNotFoundError(
                    f"Vector index not found in {self.config.chroma_dir} "
                    f"(no chroma.sqlite3). Run:  python -m src.ingest"
                )

            # Read path only. The collection must already exist -- a missing
            # index is a setup error the user must see, not something to
            # silently paper over by building it here.
            client = chromadb.PersistentClient(path=str(self.config.chroma_dir))
            try:
                self._collection = client.get_collection(self.config.chroma_collection)
            except ValueError as error:
                raise FileNotFoundError(
                    f"Collection {self.config.chroma_collection!r} not found in "
                    f"{self.config.chroma_dir}. Run:  python -m src.ingest"
                ) from error
            LOGGER.info(
                "opened collection %r (%d vectors)",
                self.config.chroma_collection,
                self._collection.count(),
            )
        return self._collection

    @property
    def embedder(self):
        # Imported here, not at module load: the Retriever must be constructible
        # (and its error messages readable) without paying for the model.
        if self._embedder is None:
            from .embedder import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    # -- search ----------------------------------------------------------

    def search(
        self,
        question: str,
        k_fetch: Optional[int] = None,
        k_context: Optional[int] = None,
        scheme_id: Optional[str] = None,
        detect: bool = True,
        min_score: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """Return up to `k_context` diverse chunks for `question`.

        An empty list is a valid, handled result -- the caller is expected to
        show a not-found message rather than retry or invent an answer.
        """
        question = (question or "").strip()
        if not question:
            return []

        fetch = k_fetch or self.k_fetch
        context = k_context or self.k_context
        threshold = self.config.min_score if min_score is None else min_score

        if detect and scheme_id is None:
            scheme_id = detect_scheme(question)

        where = {"scheme_id": scheme_id} if scheme_id else None
        LOGGER.info(
            "search: scheme=%s fetch=%d context=%d where=%s",
            scheme_id or "(unfiltered)", fetch, context, where,
        )

        vector = self.embedder.encode_one(question)
        # Reset per search: only this query's candidates are deduplicated, and
        # the dict cannot grow without bound across a long-running app.
        self._embeddings = {}
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=fetch,
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        raw = self._to_candidates(result)
        if not raw:
            LOGGER.info("search: no candidates returned")
            return []

        if threshold and threshold > 0 and raw[0].score < threshold:
            # Gate on the TOP hit, not on every candidate. Filtering each
            # candidate would silently shrink the context for a strong query
            # whose tail happens to score low. The question this answers is
            # "is anything in the corpus about this at all?" -- and if the best
            # match cannot clear the bar, nothing can. Measured spread: real
            # questions score 0.40-0.76, gibberish 0.05-0.20.
            LOGGER.info(
                "search: best score %.4f below min_score %.4f - no grounded match",
                raw[0].score, threshold,
            )
            return []

        selected = self._diversify(raw, context)
        LOGGER.info(
            "search: %d candidates -> %d selected (top score %.4f)",
            len(raw), len(selected), selected[0].score if selected else 0.0,
        )
        return selected

    # -- internals -------------------------------------------------------

    def _to_candidates(self, result) -> List[RetrievedChunk]:
        """Flatten Chroma's batch-of-one response and convert distance to score.

        Chroma returns cosine DISTANCE under a cosine space, and
        distance == 1 - cosine_similarity, so a smaller number is a better hit.
        Callers want similarity: higher is better, 1.0 is identical.
        """
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        embeddings = (result.get("embeddings") or [[]])[0]

        candidates: List[RetrievedChunk] = []
        for position, chunk_id in enumerate(ids):
            if position >= len(documents):
                break
            meta = metadatas[position] if position < len(metadatas) else {}
            distance = float(distances[position]) if position < len(distances) else 1.0
            candidates.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=documents[position],
                    scheme_id=meta.get("scheme_id", ""),
                    scheme_name=meta.get("scheme_name", ""),
                    section=meta.get("section"),
                    source_url=meta.get("source_url", ""),
                    last_updated=meta.get("last_updated", ""),
                    score=1.0 - distance,
                )
            )
            self._embeddings[chunk_id] = (
                np.asarray(embeddings[position], dtype=np.float32)
                if position < len(embeddings) and embeddings[position] is not None
                else None
            )
        return candidates

    def _diversify(
        self, candidates: Sequence[RetrievedChunk], limit: int
    ) -> List[RetrievedChunk]:
        """Drop near-duplicates, cap per-section repeats, then cap the total.

        Two passes: the first honours the per-section cap, the second backfills
        any unfilled slots without it. That way a query whose top candidates all
        come from one section still returns `limit` chunks instead of starving.
        """
        picked: List[RetrievedChunk] = []
        per_section: Dict[Optional[str], int] = {}
        deferred: List[RetrievedChunk] = []

        for candidate in candidates:
            if len(picked) >= limit:
                break
            if self._is_duplicate(candidate, picked):
                continue
            section = candidate.section
            if per_section.get(section, 0) >= self.max_per_section:
                deferred.append(candidate)
                continue
            picked.append(candidate)
            per_section[section] = per_section.get(section, 0) + 1

        if len(picked) < limit:
            for candidate in deferred:
                if len(picked) >= limit:
                    break
                if self._is_duplicate(candidate, picked):
                    continue
                picked.append(candidate)

        return picked

    def _is_duplicate(
        self, candidate: RetrievedChunk, picked: Sequence[RetrievedChunk]
    ) -> bool:
        """True if `candidate` restates something already picked.

        Uses the stored vectors when available (exact cosine). Falls back to
        identical text, so the check still works if embeddings are not returned.
        """
        vector = self._embeddings.get(candidate.chunk_id)
        for other in picked:
            if candidate.text == other.text:
                return True
            other_vector = self._embeddings.get(other.chunk_id)
            if vector is not None and other_vector is not None:
                similarity = float(np.dot(vector, other_vector))
                if similarity >= self.dedup_threshold:
                    return True
        return False


def get_retriever(**kwargs) -> Retriever:
    """Process-wide retriever, so the collection is opened once per process."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever(**kwargs)
    return _RETRIEVER


_RETRIEVER: Optional[Retriever] = None


__all__ = ["RetrievedChunk", "Retriever", "get_retriever"]
