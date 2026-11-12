-- Schema for the Insurance Policy Q&A Agent.
-- Applied by `policyqa init-db` (idempotent). {dim} is substituted with EMBEDDING_DIM.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,          -- sha256 of the file bytes (idempotent ingestion)
    filename        TEXT NOT NULL,
    title           TEXT NOT NULL,
    policy_type     TEXT NOT NULL,
    num_pages       INTEGER NOT NULL CHECK (num_pages > 0),
    embedding_model TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    content     TEXT NOT NULL,
    section     TEXT,
    page_start  INTEGER NOT NULL,
    page_end    INTEGER NOT NULL,
    flags       TEXT[] NOT NULL DEFAULT '{}',
    embedding   vector({dim}) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (
                    setweight(to_tsvector('english', coalesce(section, '')), 'A') ||
                    setweight(to_tsvector('english', content), 'B')
                ) STORED,
    UNIQUE (doc_id, ordinal)
);

-- HNSW gives sub-linear ANN search; cosine ops because vectors are L2-normalised.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc_id ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS documents_policy_type ON documents (policy_type);
