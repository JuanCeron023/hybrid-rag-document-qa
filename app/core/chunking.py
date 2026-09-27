"""Document chunking.

Splitting documents into overlapping chunks is the first and most consequential
step of a RAG pipeline: chunks that are too large dilute relevance, and chunks
with no overlap lose context at their boundaries. DocSage uses a sentence-aware
splitter with a configurable window and overlap, and tracks each chunk's source
document and character span so answers can cite exactly where they came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A retrievable slice of a document, with provenance for citations."""

    id: str
    doc_id: str
    text: str
    start: int
    end: int
    metadata: dict = field(default_factory=dict)


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    # Simple, dependency-free sentence splitter. Good enough for chunking; the
    # goal is reasonable boundaries, not linguistic perfection.
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_document(
    doc_id: str,
    text: str,
    *,
    max_chars: int = 400,
    overlap_chars: int = 80,
) -> list[Chunk]:
    """Split ``text`` into overlapping, sentence-aligned chunks.

    Sentences are packed into a chunk until ``max_chars`` is reached; the next
    chunk starts ``overlap_chars`` earlier so context spanning a boundary is not
    lost. Character offsets are preserved for citation highlighting.
    """
    sentences = _split_sentences(text)
    chunks: list[Chunk] = []
    if not sentences:
        return chunks

    buf: list[str] = []
    buf_len = 0
    cursor = 0  # running char offset into the original text
    chunk_start = 0

    def flush(end_offset: int) -> None:
        nonlocal buf, buf_len, chunk_start
        if not buf:
            return
        chunk_text = " ".join(buf)
        chunks.append(
            Chunk(
                id=f"{doc_id}::{len(chunks)}",
                doc_id=doc_id,
                text=chunk_text,
                start=chunk_start,
                end=end_offset,
            )
        )
        # Start the next buffer with a character-based overlap tail.
        if overlap_chars > 0 and len(chunk_text) > overlap_chars:
            tail = chunk_text[-overlap_chars:]
            buf = [tail]
            buf_len = len(tail)
            chunk_start = end_offset - len(tail)
        else:
            buf = []
            buf_len = 0
            chunk_start = end_offset

    for sent in sentences:
        if buf and buf_len + len(sent) + 1 > max_chars:
            flush(cursor)
        if not buf:
            chunk_start = cursor
        buf.append(sent)
        buf_len += len(sent) + 1
        cursor += len(sent) + 1

    flush(cursor)
    return chunks
