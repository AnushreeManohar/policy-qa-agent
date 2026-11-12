from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from policyqa.models import Chunk, Document


class StoreError(RuntimeError):
    """Storage backend unavailable or failed; surfaced to clients as 503 without internals."""


@dataclass
class DocumentSummary:
    doc_id: str
    filename: str
    title: str
    policy_type: str
    num_pages: int
    num_chunks: int


@dataclass
class SearchHit:
    chunk: Chunk
    title: str
    filename: str
    policy_type: str
    score: float  # cosine similarity (vector) or lexical rank score (keyword)


class VectorStore(Protocol):
    def init_schema(self) -> None: ...
    def healthcheck(self) -> bool: ...
    def has_document(self, doc_id: str) -> bool: ...
    def add_document(self, doc: Document, embeddings: np.ndarray) -> None: ...
    def delete_document(self, doc_id: str) -> bool: ...
    def list_documents(self) -> list[DocumentSummary]: ...
    def all_chunks(self) -> list[Chunk]: ...
    def vector_search(self, query_vec: np.ndarray, k: int, policy_type: str | None = None) -> list[SearchHit]: ...
    def keyword_search(self, query_terms: list[str], k: int, policy_type: str | None = None) -> list[SearchHit]: ...
    def close(self) -> None: ...
