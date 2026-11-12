"""Optional semantic layer: pretrained word vectors and part-of-speech tags from spaCy's `en_core_web_md`.

Used for two narrow jobs where pure lexical matching failed in the audit:
  1. Soft synonym matching: a question word counts as matched by an evidence word whose vector cosine is
     >= SOFT_MATCH_THRESHOLD ("monthly" ~ "month", "kitten" ~ "cat", "documents" ~ "invoice").
     The threshold is deliberately high; lower values let "weather" match "windstorm" (0.55) and
     "renters" match "tenant" (0.42), which would turn correct abstentions into wrong answers.
  2. Topic words for the evidence gate: only nouns, proper nouns and adjectives can make the agent abstain.
     Verbs/adverbs ("start", "newly") are phrasing, not topics.

Everything degrades gracefully: with NLP_MODEL=none (or the model not installed) the functions return
None/empty and the agent falls back to purely lexical behaviour.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from functools import lru_cache

import numpy as np

from policyqa.text import STOPWORDS, stem

logger = logging.getLogger(__name__)
SOFT_MATCH_THRESHOLD = 0.75
TOPIC_POS = {"NOUN", "PROPN", "ADJ"}
_WORD = re.compile(r"[a-z][a-z'-]{2,}")


class WordVectors:
    def __init__(self, nlp) -> None:  # noqa: ANN001
        self.nlp = nlp
        self._lock = threading.Lock()  # spaCy pipelines are not guaranteed thread-safe

    def vector(self, word: str) -> np.ndarray | None:
        lex = self.nlp.vocab[word]
        if not lex.has_vector:
            return None
        v = np.asarray(lex.vector, dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n else None

    def neighbors(self, word: str, candidates: set[str], threshold: float = SOFT_MATCH_THRESHOLD) -> set[str]:
        """Stems of candidate words whose vector is within `threshold` cosine of `word`."""
        q = self.vector(word)
        if q is None:
            return set()
        cands = [c for c in candidates if c != word]
        vecs = [(c, self.vector(c)) for c in cands]
        vecs = [(c, v) for c, v in vecs if v is not None]
        if not vecs:
            return set()
        mat = np.stack([v for _, v in vecs])
        sims = mat @ q
        return {stem(c) for (c, _), s in zip(vecs, sims, strict=True) if s >= threshold}

    def topic_words(self, text: str) -> set[str]:
        with self._lock:
            doc = self.nlp(text)
        return {t.lower_ for t in doc if t.pos_ in TOPIC_POS and not t.is_stop and t.is_alpha}


@lru_cache(maxsize=1)
def word_vectors() -> WordVectors | None:
    name = os.environ.get("NLP_MODEL", "en_core_web_md")
    if name.lower() in ("", "none", "off"):
        return None
    try:
        import spacy

        nlp = spacy.load(name, exclude=["parser", "ner", "lemmatizer", "senter"])
    except Exception as exc:  # noqa: BLE001 - optional dependency; fall back to lexical behaviour
        logger.warning("semantic layer disabled (%s: %s); install with: pip install -e '.[semantic]'",
                       type(exc).__name__, str(exc)[:80])
        return None
    return WordVectors(nlp)


def words_in(texts: list[str]) -> set[str]:
    out: set[str] = set()
    for t in texts:
        out.update(w for w in _WORD.findall(t.lower()) if w not in STOPWORDS)
    return out
