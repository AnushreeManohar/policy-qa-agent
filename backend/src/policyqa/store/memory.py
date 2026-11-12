"""In-process store (numpy cosine + BM25). Used by unit tests and quick local demos without Postgres."""

from __future__ import annotations

import math
import threading
from collections import Counter

import numpy as np

from policyqa.models import Chunk, Document
from policyqa.store.base import DocumentSummary, SearchHit
from policyqa.text import stem, tokens


class MemoryStore:
    def __init__(self, embedding_model: str) -> None:
        self.embedding_model = embedding_model
        self._docs: dict[str, Document] = {}
        self._vecs: dict[str, np.ndarray] = {}  # chunk_id -> vector
        self._lock = threading.Lock()

    def init_schema(self) -> None:
        return None

    def healthcheck(self) -> bool:
        return True

    def has_document(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def add_document(self, doc: Document, embeddings: np.ndarray) -> None:
        with self._lock:
            self._docs[doc.doc_id] = doc
            for chunk, vec in zip(doc.chunks, embeddings, strict=True):
                self._vecs[chunk.chunk_id] = vec

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            doc = self._docs.pop(doc_id, None)
            if doc is None:
                return False
            for c in doc.chunks:
                self._vecs.pop(c.chunk_id, None)
            return True

    def list_documents(self) -> list[DocumentSummary]:
        return [
            DocumentSummary(d.doc_id, d.filename, d.title, d.policy_type, d.num_pages, len(d.chunks))
            for d in sorted(self._docs.values(), key=lambda d: d.filename)
        ]

    def all_chunks(self) -> list[Chunk]:
        return [c for d in self._docs.values() for c in d.chunks]

    def _chunks(self, policy_type: str | None) -> list[tuple[Chunk, Document]]:
        return [
            (c, d)
            for d in self._docs.values()
            if policy_type is None or d.policy_type == policy_type
            for c in d.chunks
        ]

    def vector_search(self, query_vec: np.ndarray, k: int, policy_type: str | None = None) -> list[SearchHit]:
        pairs = self._chunks(policy_type)
        if not pairs:
            return []
        mat = np.stack([self._vecs[c.chunk_id] for c, _ in pairs])
        sims = mat @ query_vec
        order = np.argsort(-sims)[:k]
        return [SearchHit(pairs[i][0], pairs[i][1].title, pairs[i][1].filename, pairs[i][1].policy_type,
                          float(sims[i])) for i in order]

    def keyword_search(self, query_words: list[str], k: int, policy_type: str | None = None) -> list[SearchHit]:
        """BM25 over raw query words, stemmed with the same stemmer used to tokenise chunks."""
        pairs = self._chunks(policy_type)
        query_terms = [stem(w) for w in query_words]
        if not pairs or not query_terms:
            return []
        docs_toks = [tokens(f"{c.section or ''} {c.text}") for c, _ in pairs]
        n = len(docs_toks)
        avgdl = sum(len(t) for t in docs_toks) / n
        df = Counter(t for toks in docs_toks for t in set(toks))
        k1, b = 1.5, 0.75
        scores = []
        for toks in docs_toks:
            tf = Counter(toks)
            s = 0.0
            for q in set(query_terms):
                if q not in tf:
                    continue
                idf = math.log(1 + (n - df[q] + 0.5) / (df[q] + 0.5))
                s += idf * tf[q] * (k1 + 1) / (tf[q] + k1 * (1 - b + b * len(toks) / avgdl))
            scores.append(s)
        order = [i for i in np.argsort(-np.asarray(scores))[:k] if scores[i] > 0]
        return [SearchHit(pairs[i][0], pairs[i][1].title, pairs[i][1].filename, pairs[i][1].policy_type,
                          float(scores[i])) for i in order]

    def close(self) -> None:
        return None
