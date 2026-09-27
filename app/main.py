"""FastAPI application exposing DocSage."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .core.engine import RAGEngine

app = FastAPI(title="DocSage", description="Grounded document Q&A with citations")
engine = RAGEngine()

WEB_DIR = Path(__file__).resolve().parent.parent / "web" / "dist"


class Document(BaseModel):
    doc_id: str
    text: str


class Question(BaseModel):
    query: str
    top_k: int = 5
    alpha: float = 0.5  # 0 = lexical, 1 = semantic


@app.post("/api/documents")
def add_document(doc: Document) -> dict:
    if not doc.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    n = engine.add_document(doc.doc_id, doc.text)
    return {"doc_id": doc.doc_id, "chunks": n}


@app.get("/api/documents")
def list_documents() -> list[dict]:
    return engine.documents()


@app.post("/api/ask")
def ask(q: Question) -> dict:
    if not q.query.strip():
        raise HTTPException(status_code=400, detail="query is empty")
    return engine.ask(q.query, top_k=q.top_k, alpha=q.alpha)


@app.get("/api/stats")
def stats() -> dict:
    return engine.stats()


@app.on_event("startup")
def _seed() -> None:
    """Seed a small knowledge base so the demo works immediately."""
    seed_dir = Path(__file__).resolve().parent.parent / "data"
    if seed_dir.exists():
        for p in sorted(seed_dir.glob("*.txt")):
            engine.add_document(p.stem, p.read_text(encoding="utf-8"))


# Static frontend (Vite build). Assets are referenced at /assets/... by the
# built index.html, so mount that directory explicitly; the catch-all root
# serves index.html for the SPA.
if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIR / "assets")), name="assets")

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(str(WEB_DIR / "index.html"))
