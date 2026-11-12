from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, examples=["What is the windstorm deductible?"])
    policy_type: str | None = Field(None, pattern=r"^[a-z_]{2,30}$", examples=["homeowners"])
    include_trace: bool = False

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 3:
            raise ValueError("question is too short")
        return v


class CitationOut(BaseModel):
    source_id: int
    chunk_id: str
    doc_id: str
    filename: str
    title: str
    page_start: int
    page_end: int
    section: str | None
    snippet: str


class SourceOut(BaseModel):
    id: int
    chunk_id: str
    title: str
    filename: str
    section: str | None
    page_start: int
    page_end: int
    score: float
    flags: list[str]
    cited: bool
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    status: Literal["answered", "partial", "insufficient_information", "refused"]
    citations: list[CitationOut]
    sources: list[SourceOut]
    policy_type: str | None
    coverage: float | None
    missing_terms: list[str]
    faithfulness: float | None
    mode: str
    latency_ms: float
    retrieval_ms: float | None
    trace: list[dict] | None = None


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    title: str
    policy_type: str
    num_pages: int
    num_chunks: int


class IngestOut(BaseModel):
    doc_id: str
    filename: str
    title: str
    policy_type: str
    status: Literal["ingested", "duplicate"]
    num_pages: int
    num_chunks: int
    flagged_chunks: int


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    llm_mode: str
    embedding_model: str
    documents: int | None
    uploads_enabled: bool
    auth_required_for_ask: bool


class ErrorOut(BaseModel):
    error: str
    request_id: str
