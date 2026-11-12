"""Structured JSON logging with request-id correlation and secret redaction."""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
import time

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_REDACTIONS = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{10,}"), "sk-***"),
    (re.compile(r"(?i)(api[_-]?key|authorization|password|secret)(\"?\s*[:=]\s*\"?)[^\s\",]+"), r"\1\2***"),
    (re.compile(r"postgres(?:ql)?://([^:/\s]+):[^@\s]+@"), r"postgresql://\1:***@"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "***-**-****"),  # SSN-shaped values
]


def redact(text: str) -> str:
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "msg": redact(record.getMessage()),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update({k: redact(v) if isinstance(v, str) else v for k, v in extra.items()})
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(logger: logging.Logger, msg: str, **fields: object) -> None:
    logger.info(msg, extra={"fields": fields})
