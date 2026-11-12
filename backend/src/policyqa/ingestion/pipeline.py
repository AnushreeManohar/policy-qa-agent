"""PDF bytes -> validated pages -> section-aware chunks -> embeddings -> store (idempotent)."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from policyqa.config import Settings
from policyqa.embeddings import Embedder
from policyqa.ingestion.chunking import chunk_pages
from policyqa.ingestion.pdf import PdfValidationError, extract_pdf, guess_title
from policyqa.logging_setup import log_event
from policyqa.models import Document
from policyqa.store.base import VectorStore

logger = logging.getLogger(__name__)

_POLICY_TYPES = {
    "homeowners": ("homeowner", "dwelling", "ho-3", "ho3", "home"),
    "auto": ("auto", "vehicle", "motor", "car"),
    "pet": ("pet", "veterinary", "dog", "cat"),
    "health": ("health", "medical", "hmo", "ppo"),
    "travel": ("travel", "trip"),
    "life": ("life insurance", "term life", "whole life"),
}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._\- ]+")


def safe_filename(name: str) -> str:
    base = Path(name).name  # strips any path components ("../../etc/passwd")
    base = _SAFE_NAME.sub("_", base).strip(" .") or "document.pdf"
    return base[:120]


def infer_policy_type(filename: str, title: str) -> str:
    hay = f"{filename} {title}".lower()
    for ptype, keys in _POLICY_TYPES.items():
        if any(k in hay for k in keys):
            return ptype
    return "general"


@dataclass
class IngestResult:
    doc_id: str
    filename: str
    title: str
    policy_type: str
    status: str  # "ingested" | "duplicate"
    num_pages: int
    num_chunks: int
    flagged_chunks: int
    seconds: float


def ingest_bytes(
    data: bytes,
    filename: str,
    *,
    settings: Settings,
    store: VectorStore,
    embedder: Embedder,
    policy_type: str | None = None,
) -> IngestResult:
    t0 = time.perf_counter()
    filename = safe_filename(filename)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise PdfValidationError(f"File exceeds {settings.max_upload_mb} MB.")
    doc_id = hashlib.sha256(data).hexdigest()

    if store.has_document(doc_id):
        log_event(logger, "ingest.duplicate", doc_id=doc_id[:16], filename=filename)
        existing = next((d for d in store.list_documents() if d.doc_id == doc_id), None)
        return IngestResult(doc_id, filename, existing.title if existing else filename,
                            existing.policy_type if existing else "general", "duplicate",
                            existing.num_pages if existing else 0, existing.num_chunks if existing else 0,
                            0, time.perf_counter() - t0)

    pages, meta_title = extract_pdf(data, max_pages=settings.max_pdf_pages,
                                    max_chars_per_page=settings.max_chars_per_page)
    title = guess_title(pages, fallback=Path(filename).stem, meta_title=meta_title)
    ptype = (policy_type or infer_policy_type(filename, title)).lower()
    chunks = chunk_pages(doc_id, pages, size=settings.chunk_size_chars, overlap=settings.chunk_overlap_chars)
    if not chunks:
        raise PdfValidationError("No text chunks could be produced from this PDF.")

    # Contextual header: the section and title travel with the chunk text into the embedding.
    vectors = embedder.embed([f"{title} | {c.section or ''}\n{c.text}" for c in chunks])
    doc = Document(doc_id, filename, title, ptype, len(pages), chunks)
    store.add_document(doc, vectors)
    flagged = sum(1 for c in chunks if c.flags)
    result = IngestResult(doc_id, filename, title, ptype, "ingested", len(pages), len(chunks), flagged,
                          time.perf_counter() - t0)
    log_event(logger, "ingest.done", doc_id=doc_id[:16], filename=filename, pages=len(pages),
              chunks=len(chunks), flagged=flagged, seconds=round(result.seconds, 3))
    return result
