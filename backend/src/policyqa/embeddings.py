"""Embedding providers.

- `hashing`: deterministic signed feature hashing over stemmed unigrams + bigrams with sublinear TF.
  No model download, no network — a lexical baseline that makes the pipeline reproducible offline
  and in CI. It is NOT a semantic model; the README reports results with that caveat.
- `openai`: OpenAI (or compatible) embedding API, the recommended production setting.

Both return L2-normalised float32 vectors so cosine similarity == inner product in pgvector.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

import numpy as np

from policyqa.config import Settings
from policyqa.text import tokens


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = tokens(text)
            feats: dict[str, int] = {}
            for t in toks:
                feats[t] = feats.get(t, 0) + 1
            for a, b in zip(toks, toks[1:], strict=False):
                feats[f"{a}_{b}"] = feats.get(f"{a}_{b}", 0) + 1
            for feat, tf in feats.items():
                idx, sign = self._bucket(feat)
                weight = (1.0 + math.log(tf)) * (0.6 if "_" in feat else 1.0)
                out[row, idx] += sign * weight
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        from langchain_openai import OpenAIEmbeddings  # optional dependency

        if settings.openai_api_key is None:
            raise RuntimeError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
        self.dim = settings.embedding_dim
        self.name = settings.embedding_model
        self._client = OpenAIEmbeddings(
            model=settings.embedding_model,
            dimensions=settings.embedding_dim,
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.asarray(self._client.embed_documents(texts), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)


class WordVectorEmbedder:
    """Mean of pretrained 300-d word vectors (spaCy en_core_web_md, GloVe-style) over content words.
    A genuine pretrained semantic signal that runs offline; much weaker than a transformer embedder."""

    def __init__(self, settings: Settings) -> None:
        from policyqa.semantics import word_vectors, words_in

        wv = word_vectors()
        if wv is None:
            raise RuntimeError("EMBEDDING_PROVIDER=wordvec requires the semantic extra: pip install -e '.[semantic]'")
        if settings.embedding_dim != 300:
            raise RuntimeError("EMBEDDING_PROVIDER=wordvec produces 300-d vectors; set EMBEDDING_DIM=300")
        self._wv, self._words = wv, words_in
        self.dim, self.name = 300, "wordvec-en_core_web_md-300"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            vecs = [v for w in self._words([text]) if (v := self._wv.vector(w)) is not None]
            if vecs:
                m = np.mean(vecs, axis=0)
                out[i] = m / (np.linalg.norm(m) or 1.0)
        return out


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings)
    if settings.embedding_provider == "wordvec":
        return WordVectorEmbedder(settings)
    return HashingEmbedder(settings.embedding_dim)
