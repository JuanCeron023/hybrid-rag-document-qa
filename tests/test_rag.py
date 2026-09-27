"""Tests for the DocSage RAG core. These run without FastAPI or any API key,
using the dependency-free local embedder and extractive generator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.chunking import chunk_document
from app.core.engine import RAGEngine


REFUND = (
    "All ticket purchases are final. Refunds are only issued if an event is "
    "cancelled by the organizer. If an event is cancelled, ticket holders "
    "receive a full refund within 10 business days. Tickets are transferable "
    "and may be transferred for free up to 2 hours before the event."
)


def test_chunking_produces_offsets():
    chunks = chunk_document("doc", REFUND, max_chars=120, overlap_chars=30)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.doc_id == "doc"
        assert c.end >= c.start
        assert c.text


def test_retrieval_finds_relevant_answer_with_citation():
    engine = RAGEngine()
    engine.add_document("refund_policy", REFUND)
    result = engine.ask("How long do refunds take when an event is cancelled?")
    assert not result["abstained"]
    assert "10 business days" in result["answer"]
    assert result["citations"], "expected at least one citation"
    assert result["citations"][0]["doc_id"] == "refund_policy"


def test_abstains_when_no_relevant_content():
    engine = RAGEngine()
    engine.add_document("refund_policy", REFUND)
    result = engine.ask("What is the airspeed velocity of an unladen swallow?")
    assert result["abstained"] is True
    assert result["citations"] == []


def test_semantic_cache_hits_on_repeat():
    engine = RAGEngine()
    engine.add_document("refund_policy", REFUND)
    q = "Can I transfer my ticket to a friend?"
    first = engine.ask(q)
    second = engine.ask(q)
    assert first["cached"] is False
    assert second["cached"] is True
    assert engine.stats()["cache_hits"] >= 1
