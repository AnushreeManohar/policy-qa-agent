import pytest
from conftest import POLICIES, ScriptedChatModel, loaded_store, make_settings
from fastapi.testclient import TestClient

from policyqa.api.app import create_app
from policyqa.store.base import StoreError
from policyqa.store.memory import MemoryStore


def client(settings=None, store=None, chat_model=None) -> TestClient:
    settings = settings or make_settings()
    return TestClient(create_app(settings, store=store if store is not None else loaded_store(settings),
                                 chat_model=chat_model))


@pytest.fixture(scope="module")
def c():
    with client() as tc:
        yield tc


def test_health_reports_mode_and_documents(c):
    body = c.get("/api/health").json()
    assert body["status"] == "ok" and body["database"] is True
    assert body["llm_mode"] == "extractive" and body["documents"] == 4


def test_ready_when_store_is_up(c):
    assert c.get("/api/ready").json() == {"ready": True}


def test_list_documents(c):
    docs = c.get("/api/documents").json()
    assert {d["filename"] for d in docs} >= {p.name for p in POLICIES}
    assert all(d["num_chunks"] > 0 and len(d["doc_id"]) == 64 for d in docs)


def test_ask_success_returns_answer_citations_and_request_id(c):
    r = c.post("/api/ask", json={"question": "What is my collision deductible?"})
    assert r.status_code == 200 and r.headers["X-Request-ID"]
    body = r.json()
    assert body["status"] == "answered" and "$500" in body["answer"]
    assert body["citations"][0]["page_start"] >= 1 and "trace" not in body
    assert any(s["cited"] for s in body["sources"])


def test_ask_can_include_trace(c):
    body = c.post("/api/ask", json={"question": "What is my collision deductible?", "include_trace": True}).json()
    assert [t["node"] for t in body["trace"]][0] == "guard_input"


@pytest.mark.parametrize("payload", [
    {}, {"question": ""}, {"question": "  a "}, {"question": "x" * 1001},
    {"question": "valid question", "policy_type": "DROP TABLE"}, {"question": 123},
])
def test_ask_validation_errors_are_clean(c, payload):
    r = c.post("/api/ask", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert set(body) == {"error", "request_id"} and "Traceback" not in body["error"]


def test_ask_rejects_malformed_json(c):
    r = c.post("/api/ask", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_sql_like_question_is_just_text(c):
    r = c.post("/api/ask", json={"question": "'; DROP TABLE chunks; -- what is the deductible?"})
    assert r.status_code == 200
    assert c.get("/api/health").json()["documents"] == 4


def test_upload_new_duplicate_and_invalid_files():
    s = make_settings()
    with client(s, store=MemoryStore("hashing-1024")) as tc:
        data = POLICIES[0].read_bytes()
        r1 = tc.post("/api/documents", files={"file": ("p.pdf", data, "application/pdf")})
        assert r1.status_code == 201 and r1.json()["status"] == "ingested" and r1.json()["num_pages"] >= 3
        r2 = tc.post("/api/documents", files={"file": ("again.pdf", data, "application/pdf")})
        assert r2.status_code == 200 and r2.json()["status"] == "duplicate"
        r3 = tc.post("/api/documents", files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
        assert r3.status_code == 422 and r3.json()["error"] == "File is not a PDF."
        r4 = tc.post("/api/documents", files={"file": ("x.pdf", b"", "application/pdf")})
        assert r4.status_code == 422
        r5 = tc.post("/api/documents", files={"file": ("x.txt", b"hello", "text/plain")})
        assert r5.status_code == 415


def test_upload_size_limit():
    s = make_settings(max_upload_mb=1)
    with client(s, store=MemoryStore("hashing-1024")) as tc:
        big = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 10)
        r = tc.post("/api/documents", files={"file": ("big.pdf", big, "application/pdf")})
        assert r.status_code == 413


def test_uploads_can_be_disabled():
    with client(make_settings(allow_uploads=False), store=MemoryStore("x")) as tc:
        r = tc.post("/api/documents", files={"file": ("p.pdf", POLICIES[0].read_bytes(), "application/pdf")})
        assert r.status_code == 403


def test_api_key_protects_mutations_and_optionally_ask():
    s = make_settings(api_key="s3cret-key", require_api_key_for_ask=True)
    with client(s, store=MemoryStore("hashing-1024")) as tc:
        files = {"file": ("p.pdf", POLICIES[0].read_bytes(), "application/pdf")}
        assert tc.post("/api/documents", files=files).status_code == 401
        assert tc.post("/api/documents", files=files, headers={"X-API-Key": "wrong"}).status_code == 401
        ok = tc.post("/api/documents", files=files, headers={"X-API-Key": "s3cret-key"})
        assert ok.status_code == 201
        assert tc.post("/api/ask", json={"question": "What is the deductible?"}).status_code == 401
        assert tc.post("/api/ask", json={"question": "What is the deductible?"},
                       headers={"X-API-Key": "s3cret-key"}).status_code == 200
        doc_id = ok.json()["doc_id"]
        assert tc.delete(f"/api/documents/{doc_id}").status_code == 401
        assert tc.delete(f"/api/documents/{doc_id}", headers={"X-API-Key": "s3cret-key"}).status_code == 204
        assert tc.delete(f"/api/documents/{doc_id}", headers={"X-API-Key": "s3cret-key"}).status_code == 404
        assert tc.delete("/api/documents/not-a-hash", headers={"X-API-Key": "s3cret-key"}).status_code == 422


def test_delete_disabled_without_api_key(c):
    assert c.delete("/api/documents/" + "a" * 64).status_code == 403


def test_rate_limit_returns_429_with_retry_after():
    with client(make_settings(rate_limit_per_minute=2)) as tc:
        codes = [tc.post("/api/ask", json={"question": "What is the deductible?"}).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        r = tc.post("/api/ask", json={"question": "What is the deductible?"})
        assert r.status_code == 429 and int(r.headers["Retry-After"]) > 0


class BrokenStore(MemoryStore):
    def healthcheck(self):
        return False

    def vector_search(self, *a, **k):
        raise StoreError("connection refused at 10.0.0.5:5432 password=hunter2")

    keyword_search = vector_search
    list_documents = vector_search


def test_database_failure_returns_503_without_internals():
    with client(store=BrokenStore("x")) as tc:
        assert tc.get("/api/health").json()["status"] == "degraded"
        assert tc.get("/api/ready").status_code == 503
        r = tc.post("/api/ask", json={"question": "What is the deductible?"})
        assert r.status_code == 503
        assert "hunter2" not in r.text and "10.0.0.5" not in r.text
        assert tc.get("/api/documents").status_code == 503


def test_llm_failure_returns_502_without_internals():
    model = ScriptedChatModel([RuntimeError("anthropic 529 overloaded key=sk-ant-xyz")], plan=["deductible"])
    with client(make_settings(llm_provider="anthropic"), chat_model=model) as tc:
        r = tc.post("/api/ask", json={"question": "What is the collision deductible?"})
        assert r.status_code == 502 and "sk-ant" not in r.text and "overloaded" not in r.text


def test_unexpected_error_returns_generic_500():
    class Exploding(MemoryStore):
        def vector_search(self, *a, **k):
            raise ZeroDivisionError("secret internal detail")

    s = make_settings()
    with TestClient(create_app(s, store=Exploding("x")), raise_server_exceptions=False) as tc:
        r = tc.post("/api/ask", json={"question": "What is the deductible?"})
        assert r.status_code == 500 and r.json()["error"] == "Internal error." and "secret" not in r.text


def test_cors_allows_only_configured_origins(c):
    ok = c.options("/api/ask", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    bad = c.options("/api/ask", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in bad.headers


def test_security_headers_and_metrics(c):
    r = c.get("/api/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff" and r.headers["X-Frame-Options"] == "DENY"
    m = c.get("/metrics").text
    assert "policyqa_http_requests_total" in m and "policyqa_ask_latency_seconds" in m
