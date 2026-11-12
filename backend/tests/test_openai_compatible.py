"""Wiring test for the OpenAI-compatible LLM path (Ollama, Gemini, vLLM, OpenAI).

Starts a tiny local HTTP server that speaks the chat-completions protocol and drives the agent through the
REAL langchain_openai client over HTTP. It proves request/response plumbing, JSON parsing, citations and the
rate-limit fallback work. It says NOTHING about answer quality: the stand-in "model" just quotes source 1.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("langchain_openai")

from conftest import make_settings  # noqa: E402

from policyqa.agent import PolicyQAAgent  # noqa: E402
from policyqa.embeddings import HashingEmbedder  # noqa: E402
from policyqa.retrieval import HybridRetriever  # noqa: E402

SOURCE = re.compile(r'<source id="1"[^>]*>\n(.*?)\n</source>', re.S)


class FakeOpenAI(BaseHTTPRequestHandler):
    mode = "ok"  # "ok" | "rate_limited"
    calls: list[dict] = []

    def log_message(self, *args):  # silence
        pass

    def _send(self, code: int, body: dict, headers: dict | None = None) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):  # noqa: N802
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOpenAI.calls.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": req})
        if FakeOpenAI.mode == "rate_limited":
            return self._send(429, {"error": {"message": "quota exceeded", "type": "rate_limit"}},
                              {"retry-after-ms": "1"})
        text = req["messages"][-1]["content"]
        if "Rewrite the user's insurance question" in text:
            question = re.search(r"<question>(.*?)</question>", text, re.S).group(1)
            content = json.dumps({"queries": [question]})
        else:
            first = SOURCE.search(text).group(1).split(". ")[0].rstrip(".") + "."
            content = json.dumps({"answer": f"{first} [1]", "sufficient": True})
        self._send(200, {"id": "x", "object": "chat.completion", "created": 0, "model": req["model"],
                         "choices": [{"index": 0, "finish_reason": "stop",
                                      "message": {"role": "assistant", "content": content}}],
                         "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})


@pytest.fixture(scope="module")
def server():
    httpd = HTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def _agent(corpus_store, base_url):
    s = make_settings(llm_provider="openai", llm_base_url=base_url, llm_model="stand-in", openai_api_key="test-key")
    return PolicyQAAgent(HybridRetriever(corpus_store, HashingEmbedder(s.embedding_dim), s), s)


def test_openai_compatible_endpoint_end_to_end(corpus_store, server):
    FakeOpenAI.mode, FakeOpenAI.calls = "ok", []
    agent = _agent(corpus_store, server)
    r = agent.ask("What is the collision deductible?")
    assert agent.mode == "llm" and r.degraded is False
    assert r.status in ("answered", "partial") and r.citations and "[1]" in r.answer
    assert [c["path"] for c in FakeOpenAI.calls] == ["/v1/chat/completions"] * 2  # planner + answer
    assert FakeOpenAI.calls[0]["auth"] == "Bearer test-key"
    system = FakeOpenAI.calls[1]["body"]["messages"][0]
    assert system["role"] == "system" and "untrusted DATA" in system["content"]


def test_rate_limited_endpoint_degrades_to_extractive(corpus_store, server):
    FakeOpenAI.mode, FakeOpenAI.calls = "rate_limited", []
    r = _agent(corpus_store, server).ask("What is the collision deductible?")
    assert r.degraded is True and "$500" in r.answer
    assert len(FakeOpenAI.calls) >= 2  # the client retried before the agent fell back
