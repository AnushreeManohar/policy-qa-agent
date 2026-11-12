"""Integration tests against a real PostgreSQL + pgvector. Skipped unless TEST_DATABASE_URL is reachable.

These tests DROP and recreate the schema in that database — point TEST_DATABASE_URL at a throwaway DB.
"""

import numpy as np
import pytest
from conftest import POLICIES, db_url, make_settings

from policyqa.embeddings import HashingEmbedder
from policyqa.ingestion.pipeline import ingest_bytes
from policyqa.store.base import StoreError
from policyqa.store.postgres import PostgresStore

URL = db_url()
pytestmark = [pytest.mark.db, pytest.mark.skipif(URL is None, reason="TEST_DATABASE_URL not reachable")]


@pytest.fixture
def pg():
    s = make_settings(vector_store="postgres", database_url=URL)
    store = PostgresStore(s, "hashing-1024")
    store.reset()
    store.init_schema()
    yield s, store
    store.close()


def test_schema_init_is_idempotent(pg):
    _, store = pg
    store.init_schema()
    store.init_schema()
    assert store.healthcheck()


def test_ingest_search_duplicate_and_cascade_delete(pg):
    s, store = pg
    emb = HashingEmbedder(s.embedding_dim)
    r = ingest_bytes(POLICIES[0].read_bytes(), POLICIES[0].name, settings=s, store=store, embedder=emb)
    assert r.status == "ingested"
    dup = ingest_bytes(POLICIES[0].read_bytes(), "copy.pdf", settings=s, store=store, embedder=emb)
    assert dup.status == "duplicate"
    docs = store.list_documents()
    assert len(docs) == 1 and docs[0].num_chunks == r.num_chunks

    vec = store.vector_search(emb.embed(["windstorm hail deductible"])[0], 3)
    assert vec and -1.0 <= vec[0].score <= 1.0 and vec[0].chunk.page_start >= 1
    kw = store.keyword_search(["earthquake", "landslide"], 3)
    assert kw and "earthquake" in kw[0].chunk.text.lower()
    assert store.keyword_search(["earthquake"], 3, policy_type="auto") == []

    assert store.delete_document(r.doc_id)
    assert store.all_chunks() == []  # ON DELETE CASCADE removed chunks


def test_sql_metacharacters_are_inert(pg):
    s, store = pg
    ingest_bytes(POLICIES[0].read_bytes(), POLICIES[0].name, settings=s, store=store,
                 embedder=HashingEmbedder(s.embedding_dim))
    # Terms are sanitised and bound as parameters; none of these can alter the query.
    assert store.keyword_search(["'; drop table chunks; --", "x' or '1'='1"], 5) == []
    assert store.keyword_search(["deductible"], 5, policy_type="' OR 1=1 --") == []
    assert len(store.all_chunks()) > 0


@pytest.mark.parametrize("word", ["deductible", "deductibles", "eligible", "driving", "premises"])
def test_keyword_search_matches_terms_postgres_stems_differently(pg, word):
    # Audit H1: our stems ("deductibl") were stemmed AGAIN by PostgreSQL, so these never matched.
    s, store = pg
    for pdf in POLICIES:
        ingest_bytes(pdf.read_bytes(), pdf.name, settings=s, store=store, embedder=HashingEmbedder(s.embedding_dim))
    hits = store.keyword_search([word], 3)
    assert hits and word[:5] in hits[0].chunk.text.lower()


def test_memory_and_postgres_keyword_search_agree(pg):
    from policyqa.store.memory import MemoryStore

    s, store = pg
    mem = MemoryStore("hashing-1024")
    emb = HashingEmbedder(s.embedding_dim)
    for pdf in POLICIES:
        for st in (store, mem):
            ingest_bytes(pdf.read_bytes(), pdf.name, settings=s, store=st, embedder=emb)
    for words in (["deductible", "windstorm"], ["rideshare"], ["waiting", "dysplasia"]):
        assert store.keyword_search(words, 1)[0].chunk.chunk_id == mem.keyword_search(words, 1)[0].chunk.chunk_id


def test_embedding_model_mismatch_is_detected(pg):
    s, _ = pg
    other = PostgresStore(s, "some-other-embedder")
    with pytest.raises(StoreError, match="embedding model"):
        other.init_schema()


def test_unreachable_database_raises_store_error():
    s = make_settings(vector_store="postgres", database_url="postgresql://nobody:pw@127.0.0.1:1/none")
    store = PostgresStore(s, "hashing-1024")
    with pytest.raises(StoreError):
        store.init_schema()
    assert store.healthcheck() is False
    with pytest.raises(StoreError):
        store.vector_search(np.zeros(s.embedding_dim, dtype=np.float32), 3)
    store.close()
