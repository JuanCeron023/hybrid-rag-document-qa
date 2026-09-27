"""Embeddings.

DocSage supports two embedding backends behind one interface:

* ``LocalEmbedder`` — a dependency-free, deterministic embedder based on hashed
  character n-grams projected into a fixed-dimensional space and L2-normalized.
  It captures lexical/sub-word similarity well enough to demonstrate the full
  hybrid-retrieval pipeline with **zero external services or API keys**, so the
  project runs anywhere out of the box.
* ``OpenAIEmbedder`` — used automatically when ``OPENAI_API_KEY`` is set,
  producing high-quality semantic embeddings for production use.

Swapping backends changes only retrieval quality, not the surrounding code —
the clean seam a senior engineer designs for.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class LocalEmbedder:
    """Deterministic hashed n-gram embedder — no dependencies, no network."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = text.lower().split()
        # Word unigrams + character trigrams give both lexical and sub-word signal.
        grams: list[str] = list(tokens)
        for tok in tokens:
            padded = f"#{tok}#"
            grams.extend(padded[i : i + 3] for i in range(len(padded) - 2))
        for gram in grams:
            h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        return _l2_normalize(vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class OpenAIEmbedder:
    """OpenAI embeddings backend (used when OPENAI_API_KEY is present)."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI  # imported lazily so it's an optional dep

        self._client = OpenAI()
        self._model = model
        self.dim = 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]


def get_embedder() -> Embedder:
    """Return the best available embedder based on the environment."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIEmbedder()
        except Exception:
            # Fall back gracefully if the SDK isn't installed or init fails.
            pass
    return LocalEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors (assumed roughly normalized)."""
    return sum(x * y for x, y in zip(a, b))
