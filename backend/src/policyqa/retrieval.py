"""Hybrid retrieval: dense vector search + lexical full-text search, merged with Reciprocal Rank Fusion.

Why hybrid: policy questions mix vocabulary that must match exactly ("Coverage D", "waiting period",
"$500 deductible") with paraphrase ("is my dog's knee surgery paid for"). Lexical search nails the
former, vectors help with the latter; RRF combines ranks without having to calibrate their scores.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from policyqa.config import Settings
from policyqa.embeddings import Embedder
from policyqa.grounding import query_words
from policyqa.models import RetrievedChunk
from policyqa.store.base import SearchHit, VectorStore


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    latency_ms: float
    timings: dict[str, float] = field(default_factory=dict)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, settings: Settings) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings

    def retrieve(self, query: str, *, top_k: int | None = None, policy_type: str | None = None) -> RetrievalResult:
        s = self.settings
        top_k = top_k or s.retrieval_top_k
        t0 = time.perf_counter()
        qvec = self.embedder.embed([query])[0]
        t1 = time.perf_counter()
        vec_hits = self.store.vector_search(qvec, s.retrieval_candidates, policy_type)
        t2 = time.perf_counter()
        kw_hits = self.store.keyword_search(query_words(query), s.retrieval_candidates, policy_type)
        t3 = time.perf_counter()

        by_id: dict[str, SearchHit] = {}
        for h in [*vec_hits, *kw_hits]:
            by_id.setdefault(h.chunk.chunk_id, h)
        vec_rank = {h.chunk.chunk_id: i for i, h in enumerate(vec_hits, start=1)}
        kw_rank = {h.chunk.chunk_id: i for i, h in enumerate(kw_hits, start=1)}
        vec_sim = {h.chunk.chunk_id: h.score for h in vec_hits}

        fused = reciprocal_rank_fusion(
            [[h.chunk.chunk_id for h in vec_hits], [h.chunk.chunk_id for h in kw_hits]], k=s.rrf_k
        )
        best_possible = 2.0 / (s.rrf_k + 1)
        # Near-duplicate suppression (audit M2): a clause repeated across documents or endorsements otherwise
        # fills every slot with copies and crowds out other evidence. Keep the best-ranked copy only.
        # Flagged (instruction-like) chunks may never be evidence, so they must not use up evidence slots either:
        # during the audit two poisoned chunks pushed the real flood exclusion out of the top 6. They are still
        # returned AFTER the evidence, so the UI can show them as ignored.
        ranked, flagged, seen_text, dropped = [], [], set(), 0
        for cid, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
            key = " ".join(by_id[cid].chunk.text.lower().split())
            if key in seen_text:
                dropped += 1
                continue
            seen_text.add(key)
            if by_id[cid].chunk.flags:
                flagged.append((cid, score))
                continue
            ranked.append((cid, score))
            if len(ranked) == top_k:
                break
        ranked += flagged[:2]
        chunks = [
            RetrievedChunk(
                chunk=by_id[cid].chunk, title=by_id[cid].title, filename=by_id[cid].filename,
                policy_type=by_id[cid].policy_type, score=score / best_possible,
                vector_rank=vec_rank.get(cid), keyword_rank=kw_rank.get(cid), vector_sim=vec_sim.get(cid),
            )
            for cid, score in ranked
        ]
        total = (time.perf_counter() - t0) * 1000
        return RetrievalResult(chunks, total, {
            "embed_ms": (t1 - t0) * 1000, "vector_ms": (t2 - t1) * 1000, "keyword_ms": (t3 - t2) * 1000,
            "duplicates_dropped": dropped,
        })
