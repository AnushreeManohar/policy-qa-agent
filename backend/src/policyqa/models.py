"""Domain types shared by ingestion, retrieval, the agent and the API."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PageText:
    page: int  # 1-based
    text: str


@dataclass
class Chunk:
    chunk_id: str  # deterministic: f"{doc_id}:{ordinal}"
    doc_id: str
    ordinal: int
    text: str
    page_start: int
    page_end: int
    section: str | None
    flags: list[str] = field(default_factory=list)  # e.g. ["possible_injection"]


@dataclass
class Document:
    doc_id: str  # sha256 of file bytes -> idempotent re-ingestion
    filename: str
    title: str
    policy_type: str
    num_pages: int
    chunks: list[Chunk] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    title: str
    filename: str
    policy_type: str
    score: float  # fused RRF score (normalised 0..1 against best possible)
    vector_rank: int | None = None
    keyword_rank: int | None = None
    vector_sim: float | None = None


@dataclass
class Citation:
    source_id: int  # [n] marker used in the answer
    chunk_id: str
    doc_id: str
    filename: str
    title: str
    page_start: int
    page_end: int
    section: str | None
    snippet: str
