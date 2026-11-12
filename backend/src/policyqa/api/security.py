"""API-key auth and a small in-memory rate limiter.

The limiter is per-process (fine for a single container demo). Behind several replicas you would move
it to Redis or the API gateway — documented in the README's limitations.
"""

from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request, status

from policyqa.config import Settings


class RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        if self.per_minute <= 0:
            return
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                retry = int(60 - (now - q[0])) + 1
                raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded.",
                                    headers={"Retry-After": str(retry)})
            q.append(now)


def client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def check_api_key(settings: Settings, provided: str | None) -> None:
    if settings.api_key is None:
        return
    expected = settings.api_key.get_secret_value()
    if not provided or not hmac.compare_digest(provided.encode(), expected.encode()):  # constant-time
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid API key.",
                            headers={"WWW-Authenticate": "ApiKey"})


def api_key_header(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> str | None:
    return x_api_key
