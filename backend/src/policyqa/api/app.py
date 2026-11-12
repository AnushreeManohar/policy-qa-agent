"""FastAPI application: composition root, middleware, error handling and routes."""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.concurrency import run_in_threadpool

from policyqa.agent import PolicyQAAgent
from policyqa.agent.llm import LLMError
from policyqa.api.schemas import AskRequest, AskResponse, DocumentOut, HealthOut, IngestOut
from policyqa.api.security import RateLimiter, api_key_header, check_api_key, client_key
from policyqa.config import Settings, get_settings
from policyqa.embeddings import Embedder, build_embedder
from policyqa.ingestion.pdf import PdfValidationError
from policyqa.ingestion.pipeline import ingest_bytes
from policyqa.logging_setup import configure_logging, log_event, request_id_var
from policyqa.retrieval import HybridRetriever
from policyqa.store import StoreError, VectorStore, build_store

logger = logging.getLogger("policyqa.api")

REQUESTS = Counter("policyqa_http_requests_total", "HTTP requests", ["method", "route", "status"])
ASK_LATENCY = Histogram("policyqa_ask_latency_seconds", "End-to-end /ask latency",
                        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30))
RETRIEVAL_LATENCY = Histogram("policyqa_retrieval_latency_seconds", "Retrieval latency per question",
                              buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1))
ANSWER_STATUS = Counter("policyqa_answers_total", "Answers by status", ["status"])


@dataclass
class Services:
    settings: Settings
    store: VectorStore
    embedder: Embedder
    agent: PolicyQAAgent
    limiter: RateLimiter


def create_app(settings: Settings | None = None, *, store: VectorStore | None = None,
               chat_model: Any | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        embedder = build_embedder(settings)
        st = store or build_store(settings, embedder.name)
        try:
            st.init_schema()
        except StoreError as exc:
            # Start anyway: /api/health reports "degraded" and requests return 503 until the DB is back.
            logger.error("schema init failed: %s", exc)
        agent = PolicyQAAgent(HybridRetriever(st, embedder, settings), settings, chat_model=chat_model)
        app.state.services = Services(settings, st, embedder, agent, RateLimiter(settings.rate_limit_per_minute))
        log_event(logger, "startup", llm_mode=agent.mode, embedding=embedder.name, store=settings.vector_store)
        yield
        st.close()

    app = FastAPI(title="Insurance Policy Q&A Agent", version="0.1.0", lifespan=lifespan,
                  description="Grounded, cited answers over insurance policy PDFs.")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                       allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "X-API-Key",
                                                                               "X-Request-ID"])

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        rid = request.headers.get("X-Request-ID", "")[:64] or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
            route = request.scope.get("route")
            REQUESTS.labels(request.method, getattr(route, "path", "unmatched"), response.status_code).inc()
            response.headers["X-Request-ID"] = rid
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
            log_event(logger, "http", method=request.method, path=request.url.path,
                      status=response.status_code, ms=round((time.perf_counter() - t0) * 1000, 1))
            return response
        finally:
            request_id_var.reset(token)

    def _err(code: int, message: str) -> JSONResponse:
        return JSONResponse({"error": message, "request_id": request_id_var.get()}, status_code=code)

    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException) -> JSONResponse:
        resp = _err(exc.status_code, str(exc.detail))
        for k, v in (exc.headers or {}).items():
            resp.headers[k] = v
        return resp

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        return _err(422, f"Invalid request: {field} {first.get('msg', '')}".strip())

    @app.exception_handler(StoreError)
    async def _store(_: Request, exc: StoreError) -> JSONResponse:
        return _err(503, "The document store is temporarily unavailable. Please try again shortly.")

    @app.exception_handler(LLMError)
    async def _llm(_: Request, exc: LLMError) -> JSONResponse:
        logger.error("llm failure: %s", type(exc.__cause__).__name__ if exc.__cause__ else "unknown")
        return _err(502, "The language model is unavailable. Please try again shortly.")

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")  # full detail stays in server logs, never in the response
        return _err(500, "Internal error.")

    def services(request: Request) -> Services:
        return request.app.state.services

    # ---------------------------------------------------------------------------------- routes
    @app.get("/api/health", response_model=HealthOut, tags=["ops"])
    def health(svc: Services = Depends(services)) -> HealthOut:
        db_ok = svc.store.healthcheck()
        docs = None
        if db_ok:
            try:
                docs = len(svc.store.list_documents())
            except StoreError:
                db_ok = False
        return HealthOut(status="ok" if db_ok else "degraded", database=db_ok, llm_mode=svc.agent.mode,
                         embedding_model=svc.embedder.name, documents=docs,
                         uploads_enabled=svc.settings.allow_uploads,
                         auth_required_for_ask=(svc.settings.api_key is not None
                                                and svc.settings.require_api_key_for_ask))

    @app.get("/api/ready", tags=["ops"])
    def ready(svc: Services = Depends(services)) -> JSONResponse:
        """Readiness: 503 while the database is unreachable. /api/health stays 200 (liveness) so an
        orchestrator stops routing traffic during a DB outage instead of restarting healthy containers."""
        ok = svc.store.healthcheck()
        return JSONResponse({"ready": ok}, status_code=200 if ok else 503)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/documents", response_model=list[DocumentOut], tags=["documents"])
    def list_documents(svc: Services = Depends(services)) -> list[DocumentOut]:
        return [DocumentOut(**d.__dict__) for d in svc.store.list_documents()]

    @app.post("/api/documents", response_model=IngestOut, status_code=201, tags=["documents"])
    async def upload_document(request: Request, file: UploadFile = File(...),
                              policy_type: str | None = Form(None, pattern=r"^[a-z_]{2,30}$"),
                              api_key: str | None = Depends(api_key_header),
                              svc: Services = Depends(services)) -> Any:
        if not svc.settings.allow_uploads:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Uploads are disabled on this deployment.")
        check_api_key(svc.settings, api_key)
        svc.limiter.check("upload:" + client_key(request))
        if file.content_type not in ("application/pdf", "application/x-pdf", "application/octet-stream"):
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only PDF files are accepted.")
        limit = svc.settings.max_upload_mb * 1024 * 1024
        data = await file.read(limit + 1)  # never read more than the cap into memory
        if len(data) > limit:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE,
                                f"File exceeds {svc.settings.max_upload_mb} MB.")
        try:
            result = await run_in_threadpool(ingest_bytes, data, file.filename or "upload.pdf",
                                             settings=svc.settings, store=svc.store, embedder=svc.embedder,
                                             policy_type=policy_type)
        except PdfValidationError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
        body = IngestOut(**{k: v for k, v in result.__dict__.items() if k != "seconds"})
        return JSONResponse(body.model_dump(), status_code=201 if result.status == "ingested" else 200)

    @app.delete("/api/documents/{doc_id}", status_code=204, tags=["documents"])
    def delete_document(doc_id: str, api_key: str | None = Depends(api_key_header),
                        svc: Services = Depends(services)) -> Response:
        if svc.settings.api_key is None:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Deleting requires API_KEY to be configured.")
        check_api_key(svc.settings, api_key)
        if not (len(doc_id) == 64 and all(c in "0123456789abcdef" for c in doc_id)):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Invalid document id.")
        if not svc.store.delete_document(doc_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
        return Response(status_code=204)

    @app.post("/api/ask", response_model=AskResponse, response_model_exclude_none=True, tags=["qa"])
    async def ask(req: AskRequest, request: Request, api_key: str | None = Depends(api_key_header),
                  svc: Services = Depends(services)) -> AskResponse:
        if svc.settings.require_api_key_for_ask:
            check_api_key(svc.settings, api_key)
        svc.limiter.check("ask:" + client_key(request))
        if len(req.question) > svc.settings.max_question_chars:
            raise HTTPException(422, f"Question exceeds {svc.settings.max_question_chars} characters.")
        result = await run_in_threadpool(svc.agent.ask, req.question, req.policy_type)
        ASK_LATENCY.observe(result.latency_ms / 1000)
        if result.retrieval_ms is not None:
            RETRIEVAL_LATENCY.observe(result.retrieval_ms / 1000)
        ANSWER_STATUS.labels(result.status).inc()
        fields: dict[str, Any] = {"status": result.status, "mode": result.mode, "latency_ms": result.latency_ms,
                                  "citations": len(result.citations), "q_len": len(req.question)}
        if svc.settings.log_questions:
            fields["question"] = req.question
        log_event(logger, "ask", **fields)
        payload = result.to_dict()
        if not req.include_trace:
            payload["trace"] = None
        return AskResponse(**payload)

    # Serve the built React app from the same origin when present (single-container deployment).
    static_dir = Path(os.environ.get("FRONTEND_DIST", "/app/static"))
    if static_dir.is_dir() and (static_dir / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str) -> FileResponse:
            if full_path.startswith(("api/", "metrics")):
                raise HTTPException(404, "Not found.")
            candidate = (static_dir / full_path).resolve()
            if full_path and candidate.is_file() and static_dir.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(static_dir / "index.html")

    return app


def app_factory() -> FastAPI:
    return create_app()
