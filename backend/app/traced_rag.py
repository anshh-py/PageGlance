"""Traced wrappers for RAG pipeline components."""

from typing import Any, Callable
import time

from app.tracing import trace


def trace_operation(name: str, func: Callable, run_type: str = "chain", **metadata) -> Any:
    try:
        with trace(name=name, run_type=run_type, **metadata):
            result = func()
        return result
    except Exception:
        raise
