"""Hybrid retrieval: BM25 (lexical) + vector similarity (semantic).

Neither retrieval method is sufficient alone: BM25 nails exact keyword matches
but misses paraphrases; vector search captures meaning but can drift on rare
terms and identifiers. DocSage runs both, normalizes their scores, and fuses
them with a weighted sum — the standard "hybrid search" that production RAG
systems rely on. A final relevance floor lets the system *abstain* when nothing
is a good match, which is what prevents confident hallucinations.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .chunking import Chunk
from .embeddings import Embedder, cosine


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    bm25: float
    vector: float


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for",
    "and", "or", "what", "how", "when", "where", "who", "which", "do", "does",
    "can", "i", "my", "me", "you", "your", "it", "this", "that", "with", "at",
    "be", "by", "from", "as", "if", "will", "would", "should", "long", "take",
}


def _tokenize(text: str) -> list[str]:
    raw = "".join(c.lower() if c.isalnum() else " " for c in text).split()
    return [t for t in raw if t and t not in _STOPWORDS]


class HybridIndex:
    """An in-memory hybrid index over document chunks."""

    def __init__(self, embedder: Embedder, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.embedder = embedder
        self.k1 = k1
        self.b = b

        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []
        self._doc_tokens: list[list[str]] = []
        self._df: Counter[str] = Counter()
        self._avg_len: float = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vecs = self.embedder.embed([c.text for c in chunks])
        for c, v in zip(chunks, vecs):
            toks = _tokenize(c.text)
            self.chunks.append(c)
            self.vectors.append(v)
            self._doc_tokens.append(toks)
            for term in set(toks):
                self._df[term] += 1
        total = sum(len(t) for t in self._doc_tokens)
        self._avg_len = total / max(1, len(self._doc_tokens))

    # --- scoring ------------------------------------------------------------

    def _bm25(self, query_terms: list[str], doc_idx: int) -> float:
        toks = self._doc_tokens[doc_idx]
        if not toks:
            return 0.0
        tf = Counter(toks)
        dl = len(toks)
        n = len(self._doc_tokens)
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            df = self._df.get(term, 0) or 1
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self._avg_len or 1))
            score += idf * (freq * (self.k1 + 1)) / (denom or 1)
        return score

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        alpha: float = 0.5,
        min_score: float = 0.12,
    ) -> list[ScoredChunk]:
        """Return the top_k chunks by fused score.

        ``alpha`` weights vector vs. BM25 (0 = pure lexical, 1 = pure semantic).
        Results below ``min_score`` after fusion are dropped, enabling
        abstention when the corpus has nothing relevant.

        Scores are squashed into [0, 1] with bounded functions rather than
        min-max normalization: min-max collapses to zero when there is a single
        candidate or when all candidates tie, which breaks retrieval on small
        corpora. Cosine is already in [0, 1] for non-negative vectors; BM25 is
        squashed with ``x / (x + k)`` so it saturates smoothly.
        """
        if not self.chunks:
            return []

        q_terms = _tokenize(query)
        q_vec = self.embedder.embed([query])[0]

        scored: list[ScoredChunk] = []
        for i, chunk in enumerate(self.chunks):
            bm25 = _squash(self._bm25(q_terms, i))
            vec = max(0.0, cosine(q_vec, self.vectors[i]))
            fused = alpha * vec + (1 - alpha) * bm25
            scored.append(ScoredChunk(chunk=chunk, score=fused, bm25=bm25, vector=vec))

        scored.sort(key=lambda s: s.score, reverse=True)
        top = [s for s in scored[:top_k] if s.score >= min_score]
        return top


def _squash(x: float, k: float = 3.0) -> float:
    """Map a non-negative score into [0, 1) with smooth saturation."""
    if x <= 0:
        return 0.0
    return x / (x + k)
