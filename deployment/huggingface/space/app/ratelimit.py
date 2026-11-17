"""Server-side rate limiting for ingestion and LLM endpoints.

The Hugging Face deployment does not provide Redis, so limits are enforced with
a fixed-window counter stored in the same shared PostgreSQL/Neon database used
by the application. That store is shared across uvicorn workers and survives
restarts, so the limits are production-compatible without extra infrastructure.

Identity: limits are keyed on the verified installation id from M9.2 (never a
frontend-supplied value), with the client IP address used as a secondary key.
The IP key is only used as a coarse per-IP backstop for behaviour like bursts of
registrations; the installation key is authoritative for owned workflows.

Failure policy: if the rate-limit store is unreachable we fail CLOSED and
return HTTP 503 Service Unavailable after logging a critical warning. We never
silently allow unlimited (potentially expensive) requests when the limit
cannot be enforced, because an unthrottled flood would otherwise be able to
bypass the quota and drive up cost/load. The deployment shares the same
PostgreSQL store with the application, so when that store is down the
ingestion/LLM work would fail independently anyway; returning 503 keeps the
behaviour safe, observable and consistent.
"""

import logging
import math
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import HTTPException, Request
from sqlalchemy import select

from app.db import create_session_factory

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = 60
DEFAULT_LIMITS = {
    "upload": 20,
    "ingest_url": 20,
    "ask": 60,
    "ask_stream": 60,
    "register": 10,
    "stats": 120,
}
DEFAULT_IP_LIMIT = 120
ENV_LIMIT_SUFFIX = "_RATE_LIMIT_MAX"
ENV_WINDOW = "PAGEGLANCE_RATE_LIMIT_WINDOW_SECONDS"
ENV_ENABLED = "PAGEGLANCE_RATE_LIMIT_ENABLED"


def _load_env() -> None:
    env_file = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_file, override=False)


def get_rate_limit_window_seconds() -> int:
    _load_env()
    try:
        raw = os.getenv(ENV_WINDOW, str(DEFAULT_WINDOW_SECONDS)).strip()
        value = int(raw)
    except ValueError:
        return DEFAULT_WINDOW_SECONDS
    return value if math.isfinite(value) and value >= 1 else DEFAULT_WINDOW_SECONDS


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def is_rate_limiting_enabled() -> bool:
    _load_env()
    return _env_bool(ENV_ENABLED, True)


def get_action_limit(action: str) -> int:
    _load_env()
    if action == "ip":
        raw = os.getenv("PAGEGLANCE_IP_RATE_LIMIT_MAX", str(DEFAULT_IP_LIMIT)).strip()
        default = DEFAULT_IP_LIMIT
    else:
        raw = os.getenv(action.upper() + ENV_LIMIT_SUFFIX, "").strip()
        default = DEFAULT_LIMITS.get(action, 60)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _counter_key(action: str, key: str, window_start: int) -> tuple[str, str, int]:
    return action, key, window_start


class RateLimiter:
    """Fixed-window rate limiting backed by the shared application database."""

    def __init__(self, session_factory=None) -> None:
        # Lazy factory: calling self._session_factory() returns a sessionmaker,
        # and calling that () yields a Session. Mirrors app.auth.get_session_factory.
        self._session_factory = session_factory or create_session_factory

    def _get_counter(self, session, action: str, key: str, window_start: int):
        from app.models import RateLimitCounter

        return session.scalar(
            select(RateLimitCounter).where(
                RateLimitCounter.action == action,
                RateLimitCounter.key == key,
                RateLimitCounter.window_start == window_start,
            )
        )

    def _increment(self, session, action: str, key: str, window_start: int) -> int:
        from app.models import RateLimitCounter

        counter = self._get_counter(session, action, key, window_start)
        if counter is None:
            counter = RateLimitCounter(
                action=action, key=key, window_start=window_start, count=1
            )
            session.add(counter)
            session.commit()
            return 1
        counter.count += 1
        session.commit()
        return counter.count

    def check(self, action: str, key: str, *, window_seconds: int | None = None) -> None:
        """Increment the window counter for (action, key) and reject over limits."""
        window_seconds = window_seconds or get_rate_limit_window_seconds()
        limit = get_action_limit(action)
        if limit <= 0:
            return
        now = int(time.time())
        window_start = now - (now % window_seconds)
        try:
            with self._session_factory()() as session:
                count = self._increment(session, action, key, window_start)
        except Exception as error:
            logger.critical(
                "rate_limit_store_unavailable action=%s fail_closed=true error_type=%s",
                action,
                type(error).__name__,
            )
            raise HTTPException(
                status_code=503,
                detail="Rate limiting is temporarily unavailable. Please try again shortly.",
            ) from None
        if count > limit:
            retry_after = window_start + window_seconds - now
            logger.info(
                "rate_limit_exceeded action=%s key_len=%d count=%d limit=%d retry_after=%d",
                action,
                len(key),
                count,
                limit,
                retry_after,
            )
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again shortly.",
                headers={"Retry-After": str(max(retry_after, 1))},
            )

    def check_installation(self, installation_id: str, action: str) -> None:
        self.check(action, f"inst:{installation_id}")

    def check_ip(self, request: Request, action: str = "ip") -> None:
        forwarded = request.headers.get("x-forwarded-for", "")
        ip = (forwarded.split(",")[0].strip() if forwarded else request.client.host if request.client else "unknown")
        self.check(action, f"ip:{ip}")
