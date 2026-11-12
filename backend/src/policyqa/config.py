"""Central configuration. Every tunable comes from the environment (12-factor); nothing is hardcoded."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Storage ---------------------------------------------------------------------------------
    database_url: str = "postgresql://policyqa:policyqa@localhost:5432/policyqa"
    vector_store: Literal["postgres", "memory"] = "postgres"
    db_pool_min: int = 1
    db_pool_max: int = 5

    # --- Embeddings ------------------------------------------------------------------------------
    # "hashing" is a deterministic, dependency-free baseline so the project runs fully offline.
    embedding_provider: Literal["hashing", "wordvec", "openai"] = "hashing"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1024

    # --- LLM -------------------------------------------------------------------------------------
    # "extractive" never calls a model: answers are assembled from retrieved sentences (offline/CI mode).
    llm_provider: Literal["extractive", "anthropic", "openai"] = "extractive"
    llm_model: str = "claude-sonnet-4-5"
    llm_base_url: str | None = None  # OpenAI-compatible endpoints (e.g. Ollama, vLLM)
    llm_temperature: float = 0.0
    llm_timeout_s: float = 60.0
    # On LLM failure, answer in extractive mode (flagged as degraded) instead of returning 502.
    llm_fallback_extractive: bool = True
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    # --- Chunking / retrieval --------------------------------------------------------------------
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200
    retrieval_top_k: int = 6
    retrieval_candidates: int = 20  # per-retriever candidate pool before rank fusion
    rrf_k: int = 60
    # Share of the question's specific terms that must appear in the retrieved evidence; below this
    # the agent broadens the search once and then abstains rather than guess.
    min_term_coverage: float = 0.5
    max_retrieval_attempts: int = 2
    min_sentence_support: float = 0.6  # verification threshold for LLM-generated sentences
    max_regenerations: int = 1

    # --- Ingestion safety ------------------------------------------------------------------------
    max_upload_mb: int = 15
    max_pdf_pages: int = 300
    max_chars_per_page: int = 20_000

    # --- API / security --------------------------------------------------------------------------
    api_key: SecretStr | None = None  # if set, mutating endpoints (and optionally /ask) require X-API-Key
    require_api_key_for_ask: bool = False
    allow_uploads: bool = True
    # Audit C2: without API_KEY, uploads used to be open to anyone, letting an anonymous user add a fake
    # "policy" that the agent then quotes as fact. Anonymous uploads now require an explicit opt-in.
    allow_anonymous_uploads: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://localhost:8080"])
    rate_limit_per_minute: int = 30
    max_question_chars: int = 1000
    log_level: str = "INFO"
    log_questions: bool = False  # off by default: questions may contain personal data

    @field_validator("api_key", "anthropic_api_key", "openai_api_key", "llm_base_url", mode="before")
    @classmethod
    def _empty_is_unset(cls, v: object) -> object:
        # Audit finding: `API_KEY=` (empty, as in .env.example) used to become a key nobody could supply.
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                import json

                return json.loads(v)
            return [o.strip() for o in v.split(",") if o.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
