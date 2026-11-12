from policyqa.store.base import DocumentSummary, SearchHit, StoreError, VectorStore

__all__ = ["DocumentSummary", "SearchHit", "StoreError", "VectorStore", "build_store"]


def build_store(settings, embedding_model: str) -> VectorStore:  # type: ignore[no-untyped-def]
    if settings.vector_store == "memory":
        from policyqa.store.memory import MemoryStore

        return MemoryStore(embedding_model)
    from policyqa.store.postgres import PostgresStore

    return PostgresStore(settings, embedding_model)
