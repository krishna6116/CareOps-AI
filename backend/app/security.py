"""Access control for write and expensive endpoints.

- Administrative endpoints (rerunning the pipeline, recording alerts,
  generating recommendations) require ``X-API-Key`` to match
  ``CAREOPS_API_KEY``. If no key is configured they are allowed only when
  ``CAREOPS_ENVIRONMENT=development``, so a misconfigured deployment fails
  closed.
- Interactive endpoints (what-if simulation, assistant) stay open for the
  dashboard but are rate limited per client address, because each request
  runs a solver.

The limiter is in-process: it protects a single API instance, which is how
this project is deployed. A multi-instance deployment would need a shared
store.
"""

from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

from backend.app.config import settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    configured = settings.api_key
    if not configured:
        if settings.environment == "development":
            return
        raise HTTPException(
            status_code=503,
            detail="This endpoint is disabled: CAREOPS_API_KEY is not configured.",
        )
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), configured.encode()):
        raise HTTPException(status_code=401, detail="A valid X-API-Key header is required.")


class RateLimiter:
    """Sliding-window limit of ``limit`` requests per ``window_seconds`` per client."""

    def __init__(self, limit: int, window_seconds: float = 60.0):
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> float | None:
        """Record a hit; return seconds to wait if the limit is exceeded."""
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return self.window - (now - hits[0])
            hits.append(now)
            return None

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


interactive_limiter = RateLimiter(settings.interactive_rate_limit_per_minute)


def rate_limit_interactive(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    retry_after = interactive_limiter.check(client)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Too many simulation or assistant requests; try again shortly.",
            headers={"Retry-After": str(max(1, int(retry_after + 0.5)))},
        )
