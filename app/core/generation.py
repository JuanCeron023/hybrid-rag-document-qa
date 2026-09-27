"""Answer generation with citations and abstention.

The generator turns retrieved chunks into a grounded answer. Two backends:

* ``ExtractiveGenerator`` (default, no API key) — selects and stitches the most
  query-relevant sentences from the retrieved chunks and attaches citations.
  It never invents content, so it cannot hallucinate; if retrieval returned
  nothing, it abstains.
* ``LLMGenerator`` (when ``OPENAI_API_KEY`` is set) — prompts an LLM with the
  retrieved context and strict instructions to answer *only* from that context
  and to say "I don't know" otherwise.

Both enforce the same contract: every answer is either grounded in cited
sources or an explicit abstention. That contract is the whole point of RAG.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .retrieval import ScoredChunk

ABSTAIN = "I couldn't find an answer to that in the provided documents."


@dataclass
class Answer:
    text: str
    citations: list[dict]
    abstained: bool


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _overlap(query: str, sentence: str) -> int:
    q = set(w.lower() for w in re.findall(r"\w+", query))
    s = set(w.lower() for w in re.findall(r"\w+", sentence))
    return len(q & s)


class ExtractiveGenerator:
    """Grounded, non-hallucinating extractive answerer."""

    def generate(self, query: str, hits: list[ScoredChunk]) -> Answer:
        if not hits:
            return Answer(text=ABSTAIN, citations=[], abstained=True)

        # Rank sentences from the top chunks by query-term overlap.
        candidates: list[tuple[int, str, ScoredChunk]] = []
        for hit in hits:
            for sent in _sentences(hit.chunk.text):
                candidates.append((_overlap(query, sent), sent, hit))
        candidates.sort(key=lambda c: c[0], reverse=True)

        chosen = [c for c in candidates if c[0] > 0][:3]
        if not chosen:
            return Answer(text=ABSTAIN, citations=[], abstained=True)

        answer_text = " ".join(sent for _, sent, _ in chosen)
        citations = _dedupe_citations(hit for _, _, hit in chosen)
        return Answer(text=answer_text, citations=citations, abstained=False)


class LLMGenerator:
    """OpenAI-backed generator constrained to the retrieved context."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model

    def generate(self, query: str, hits: list[ScoredChunk]) -> Answer:
        if not hits:
            return Answer(text=ABSTAIN, citations=[], abstained=True)
        context = "\n\n".join(
            f"[{i+1}] (from {h.chunk.doc_id}) {h.chunk.text}" for i, h in enumerate(hits)
        )
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, reply exactly: "
            f'"{ABSTAIN}". Cite sources inline like [1], [2].\n\n'
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        abstained = ABSTAIN.lower() in text.lower()
        citations = [] if abstained else _dedupe_citations(h for h in hits)
        return Answer(text=text, citations=citations, abstained=abstained)


def _dedupe_citations(hits) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for h in hits:
        if h.chunk.id in seen:
            continue
        seen.add(h.chunk.id)
        out.append(
            {
                "chunk_id": h.chunk.id,
                "doc_id": h.chunk.doc_id,
                "snippet": h.chunk.text[:240],
                "score": round(h.score, 3),
            }
        )
    return out


def get_generator():
    """Return the LLM generator when configured, else the extractive one."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            return LLMGenerator()
        except Exception:
            pass
    return ExtractiveGenerator()
