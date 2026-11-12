"""PostgreSQL + pgvector store.

All SQL is parameterised (psycopg placeholders); user text never reaches SQL as code. The keyword
query is built from our own sanitised tokens and passed as a bound parameter to to_tsquery.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout

from policyqa.config import Settings
from policyqa.models import Chunk, Document
from policyqa.store.base import DocumentSummary, SearchHit, StoreError

logger = logging.getLogger(__name__)
_SAFE_TERM = re.compile(r"^[a-z0-9]{2,40}$")

_SELECT = """
SELECT c.chunk_id, c.doc_id, c.ordinal, c.content, c.section, c.page_start, c.page_end, c.flags,
       d.title, d.filename, d.policy_type, {score} AS score
FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
"""


def _row_to_hit(row: dict) -> SearchHit:
    chunk = Chunk(
        chunk_id=row["chunk_id"], doc_id=row["doc_id"], ordinal=row["ordinal"], text=row["content"],
        page_start=row["page_start"], page_end=row["page_end"], section=row["section"],
        flags=list(row["flags"] or []),
    )
    return SearchHit(chunk, row["title"], row["filename"], row["policy_type"], float(row["score"]))


class PostgresStore:
    def __init__(self, settings: Settings, embedding_model: str) -> None:
        self.settings = settings
        self.embedding_model = embedding_model
        self._pool: ConnectionPool | None = None

    # -- connection management ------------------------------------------------------------------
    @property
    def pool(self) -> ConnectionPool:
        if self._pool is None:
            self._pool = ConnectionPool(
                self.settings.database_url,
                min_size=self.settings.db_pool_min,
                max_size=self.settings.db_pool_max,
                kwargs={"row_factory": dict_row, "connect_timeout": 5},
                configure=self._configure,
                open=True,
                timeout=5,
            )
        return self._pool

    @staticmethod
    def _configure(conn: psycopg.Connection) -> None:
        try:
            register_vector(conn)
        except psycopg.ProgrammingError:
            # Extension not created yet (fresh DB before init_schema); init_schema re-registers.
            conn.rollback()

    @contextmanager
    def _conn(self) -> Iterator[psycopg.Connection]:
        try:
            with self.pool.connection() as conn:
                yield conn
        except (psycopg.OperationalError, psycopg.InterfaceError, PoolTimeout) as exc:
            logger.error("database error: %s", type(exc).__name__)
            raise StoreError("database unavailable") from exc

    # -- schema ---------------------------------------------------------------------------------
    def init_schema(self) -> None:
        sql = resources.files("policyqa.store").joinpath("schema.sql").read_text()
        sql = sql.replace("{dim}", str(self.settings.embedding_dim))
        try:
            with psycopg.connect(self.settings.database_url, connect_timeout=5, autocommit=True) as conn:
                conn.execute(sql)
                register_vector(conn)
                row = conn.execute("SELECT value FROM schema_meta WHERE key = 'embedding_model'").fetchone()
                if row is None:
                    conn.execute(
                        "INSERT INTO schema_meta (key, value) VALUES ('embedding_model', %s), ('embedding_dim', %s)",
                        (self.embedding_model, str(self.settings.embedding_dim)),
                    )
                elif row[0] != self.embedding_model:
                    raise StoreError(
                        f"Database was built with embedding model '{row[0]}' but the app is configured for "
                        f"'{self.embedding_model}'. Re-index (policyqa reset-db && policyqa ingest) or change config."
                    )
        except psycopg.OperationalError as exc:
            raise StoreError("database unavailable") from exc
        # Pool connections opened before the extension existed must be recycled.
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def reset(self) -> None:
        with psycopg.connect(self.settings.database_url, autocommit=True) as conn:
            conn.execute("DROP TABLE IF EXISTS chunks, documents, schema_meta CASCADE")

    def healthcheck(self) -> bool:
        try:
            with self._conn() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except Exception:  # noqa: BLE001
            return False

    # -- writes ---------------------------------------------------------------------------------
    def has_document(self, doc_id: str) -> bool:
        with self._conn() as conn:
            return conn.execute("SELECT 1 FROM documents WHERE doc_id = %s", (doc_id,)).fetchone() is not None

    def add_document(self, doc: Document, embeddings: np.ndarray) -> None:
        rows = [
            (c.chunk_id, doc.doc_id, c.ordinal, c.text, c.section, c.page_start, c.page_end, c.flags, vec)
            for c, vec in zip(doc.chunks, embeddings, strict=True)
        ]
        with self._conn() as conn, conn.transaction():  # all-or-nothing: no half-ingested documents
            conn.execute(
                "INSERT INTO documents (doc_id, filename, title, policy_type, num_pages, embedding_model) "
                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (doc_id) DO NOTHING",
                (doc.doc_id, doc.filename, doc.title, doc.policy_type, doc.num_pages, self.embedding_model),
            )
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (chunk_id, doc_id, ordinal, content, section, page_start, page_end, "
                    "flags, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    rows,
                )

    def delete_document(self, doc_id: str) -> bool:
        with self._conn() as conn:
            return conn.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,)).rowcount > 0

    def list_documents(self) -> list[DocumentSummary]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT d.doc_id, d.filename, d.title, d.policy_type, d.num_pages, count(c.chunk_id) AS n "
                "FROM documents d LEFT JOIN chunks c ON c.doc_id = d.doc_id "
                "GROUP BY d.doc_id ORDER BY d.filename"
            ).fetchall()
        return [DocumentSummary(r["doc_id"], r["filename"], r["title"], r["policy_type"], r["num_pages"], r["n"])
                for r in rows]

    def all_chunks(self) -> list[Chunk]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chunk_id, doc_id, ordinal, content, section, page_start, page_end, flags "
                "FROM chunks ORDER BY doc_id, ordinal"
            ).fetchall()
        return [Chunk(r["chunk_id"], r["doc_id"], r["ordinal"], r["content"], r["page_start"], r["page_end"],
                      r["section"], list(r["flags"] or [])) for r in rows]

    # -- reads ----------------------------------------------------------------------------------
    def vector_search(self, query_vec: np.ndarray, k: int, policy_type: str | None = None) -> list[SearchHit]:
        where, params = ("WHERE d.policy_type = %(pt)s", {"pt": policy_type}) if policy_type else ("", {})
        sql = (_SELECT.format(score="1 - (c.embedding <=> %(q)s)") + where
               + " ORDER BY c.embedding <=> %(q)s LIMIT %(k)s")
        with self._conn() as conn:
            rows = conn.execute(sql, {"q": np.asarray(query_vec, dtype=np.float32), "k": k, **params}).fetchall()
        return [_row_to_hit(r) for r in rows]

    def keyword_search(self, query_words: list[str], k: int, policy_type: str | None = None) -> list[SearchHit]:
        """Full-text search over RAW words: PostgreSQL's English stemmer is applied by to_tsquery, matching
        how to_tsvector indexed the chunks. Words are restricted to [a-z0-9] and bound as a parameter."""
        terms = sorted({t for t in (w.lower() for w in query_words) if _SAFE_TERM.match(t)})
        if not terms:
            return []
        tsquery = " | ".join(terms)  # OR semantics: natural-language questions rarely match every term
        where = "WHERE c.tsv @@ to_tsquery('english', %(tq)s)"
        params: dict = {"tq": tsquery, "k": k}
        if policy_type:
            where += " AND d.policy_type = %(pt)s"
            params["pt"] = policy_type
        sql = (_SELECT.format(score="ts_rank_cd(c.tsv, to_tsquery('english', %(tq)s), 32)") + where
               + " ORDER BY score DESC LIMIT %(k)s")
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_hit(r) for r in rows]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None
