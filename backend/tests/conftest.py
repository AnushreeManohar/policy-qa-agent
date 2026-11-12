from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from policyqa.agent import PolicyQAAgent
from policyqa.config import Settings
from policyqa.embeddings import HashingEmbedder
from policyqa.ingestion.pipeline import ingest_bytes
from policyqa.retrieval import HybridRetriever
from policyqa.store.memory import MemoryStore

ROOT = Path(__file__).resolve().parents[2]
POLICIES = sorted((ROOT / "data" / "policies").glob("*.pdf"))
ADVERSARIAL = ROOT / "data" / "adversarial" / "coastal_flood_supplement.pdf"


@pytest.fixture(scope="session", autouse=True)
def _sample_pdfs_exist() -> None:
    if not POLICIES or not ADVERSARIAL.exists():
        pytest.exit("Sample PDFs missing: run `python scripts/generate_sample_policies.py` first.", 2)


def make_settings(**overrides) -> Settings:
    base = dict(vector_store="memory", llm_provider="extractive", embedding_provider="hashing",
                api_key=None, rate_limit_per_minute=1000, cors_origins=["http://localhost:5173"])
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


def loaded_store(settings: Settings, include_adversarial: bool = True) -> MemoryStore:
    store = MemoryStore("hashing-1024")
    emb = HashingEmbedder(settings.embedding_dim)
    for pdf in [*POLICIES, *([ADVERSARIAL] if include_adversarial else [])]:
        ingest_bytes(pdf.read_bytes(), pdf.name, settings=settings, store=store, embedder=emb)
    return store


@pytest.fixture(scope="module")
def corpus_store() -> MemoryStore:
    return loaded_store(make_settings())


@pytest.fixture(scope="module")
def agent(corpus_store: MemoryStore) -> PolicyQAAgent:
    s = make_settings()
    return PolicyQAAgent(HybridRetriever(corpus_store, HashingEmbedder(s.embedding_dim), s), s)


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class ScriptedChatModel:
    """Stands in for a chat model: planner prompts (plain strings) get a JSON plan; answer prompts
    (message lists) get the next scripted reply. Records every answer prompt for assertions."""

    def __init__(self, answers: list[str | Exception], plan: list[str] | None = None) -> None:
        self.answers = list(answers)
        self.plan = plan
        self.answer_prompts: list[list] = []

    def invoke(self, messages):  # noqa: ANN001
        if isinstance(messages, str):
            if self.plan is None:
                return FakeMessage("not json")  # exercises the heuristic planner fallback
            return FakeMessage(json.dumps({"queries": self.plan}))
        self.answer_prompts.append(messages)
        nxt = self.answers.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return FakeMessage(nxt)


def db_url() -> str | None:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        return None
    try:
        import psycopg

        with psycopg.connect(url, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        return url
    except Exception:  # noqa: BLE001
        return None
