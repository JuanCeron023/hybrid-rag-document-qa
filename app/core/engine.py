"""The RAG engine: ties chunking, indexing, retrieval, semantic caching, and
grounded generation into one service.

A semantic cache short-circuits repeated/similar questions: if a new query is
very close (by embedding cosine similarity) to a previously answered one, the
cached answer is returned instantly. This is a real production optimization —
it cuts latency and LLM cost for the long tail of near-duplicate questions.
"""

from __future__ import annotations

from dataclasses import dataclass

from .chunking import chunk_document
from .embeddings import Embedder, cosine, get_embedder
from .generation import Answer, get_generator
from .retrieval import HybridIndex


@dataclass
class _CacheEntry:
    vector: list[float]
    answer: Answer


class RAGEngine:
    def __init__(self, *, cache_threshold: float = 0.95) -> None:
        self.embedder: Embedder = get_embedder()
        self.index = HybridIndex(self.embedder)
        self.generator = get_generator()
        self._docs: dict[str, str] = {}
        self._cache: list[_CacheEntry] = []
        self._cache_threshold = cache_threshold
        self._cache_hits = 0

    # --- ingestion ----------------------------------------------------------

    def add_document(self, doc_id: str, text: str) -> int:
        """Chunk, index, and store a document. Returns the chunk count."""
        self._docs[doc_id] = text
        chunks = chunk_document(doc_id, text)
        self.index.add(chunks)
        # Any new knowledge may change answers, so invalidate the cache.
        self._cache.clear()
        return len(chunks)

    # --- query --------------------------------------------------------------

    def ask(self, query: str, *, top_k: int = 5, alpha: float = 0.5) -> dict:
        q_vec = self.embedder.embed([query])[0]

        # Semantic cache lookup.
        for entry in self._cache:
            if cosine(q_vec, entry.vector) >= self._cache_threshold:
                self._cache_hits += 1
                return _to_dict(entry.answer, cached=True)

        hits = self.index.search(query, top_k=top_k, alpha=alpha)
        answer = self.generator.generate(query, hits)

        if not answer.abstained:
            self._cache.append(_CacheEntry(vector=q_vec, answer=answer))
        return _to_dict(answer, cached=False)

    # --- introspection ------------------------------------------------------

    def stats(self) -> dict:
        return {
            "documents": len(self._docs),
            "chunks": len(self.index.chunks),
            "cache_entries": len(self._cache),
            "cache_hits": self._cache_hits,
            "embedder": type(self.embedder).__name__,
            "generator": type(self.generator).__name__,
        }

    def documents(self) -> list[dict]:
        return [{"doc_id": d, "chars": len(t)} for d, t in self._docs.items()]


def _to_dict(answer: Answer, *, cached: bool) -> dict:
    return {
        "answer": answer.text,
        "abstained": answer.abstained,
        "citations": answer.citations,
        "cached": cached,
    }
