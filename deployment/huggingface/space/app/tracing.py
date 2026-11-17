"""LangSmith tracing utilities for PageGlance.

Configures and provides tracing decorators/helpers. Disabled when LANGSMITH_TRACING=false.
"""

import os
import functools
from typing import Any, Callable, Optional

_TRACING_ENABLED = None


def _is_tracing_enabled() -> bool:
    global _TRACING_ENABLED
    if _TRACING_ENABLED is not None:
        return _TRACING_ENABLED
    tracing = os.getenv("LANGSMITH_TRACING", "").strip().lower()
    if tracing in {"false", "0", "no", "off"}:
        _TRACING_ENABLED = False
        return False
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        _TRACING_ENABLED = False
        return False
    _TRACING_ENABLED = tracing in {"true", "1", "yes", "on"} or bool(api_key)
    return _TRACING_ENABLED


def get_client():
    if not _is_tracing_enabled():
        return None
    try:
        from langsmith import Client
        return Client()
    except Exception:
        return None


def traceable(
    name: Optional[str] = None,
    run_type: str = "chain",
    **kwargs,
):
    def decorator(func: Callable) -> Callable:
        if not _is_tracing_enabled():
            return func
        try:
            from langsmith import traceable as ls_traceable
            return ls_traceable(name=name or func.__name__, run_type=run_type, **kwargs)(func)
        except Exception:
            return func

    if callable(name):
        func = name
        name = None
        return decorator(func)
    return decorator


class trace:
    def __init__(self, name: str, run_type: str = "chain", **metadata):
        self.name = name
        self.run_type = run_type
        self.metadata = metadata
        self.run = None

    def __enter__(self):
        if not _is_tracing_enabled():
            return self
        try:
            from langsmith.run_helpers import trace as ls_trace
            self._ctx = ls_trace(name=self.name, run_type=self.run_type, **self.metadata)
            self.run = self._ctx.__enter__()
            return self.run
        except Exception:
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not _is_tracing_enabled() or not hasattr(self, '_ctx'):
            return False
        try:
            return self._ctx.__exit__(exc_type, exc_val, exc_tb)
        except Exception:
            return False


def trace_run(func: Callable, name: str = None, run_type: str = "chain", **metadata) -> Any:
    if not _is_tracing_enabled():
        return func()
    try:
        from langsmith.run_helpers import trace
        with trace(name=name or getattr(func, '__name__', 'operation'), run_type=run_type, **metadata):
            return func()
    except Exception:
        return func()
