
"""Production LangSmith tracing integration for PageGlance."""

import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

_TRACING_ENABLED = None


def is_tracing_enabled() -> bool:
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

    _TRACING_ENABLED = tracing in {"true", "1", "yes", "on"} or True
    return _TRACING_ENABLED


def configure() -> bool:
    if not is_tracing_enabled():
        return False

    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()

    if not api_key:
        return False

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
    os.environ.setdefault(
        "LANGCHAIN_PROJECT",
        os.getenv("LANGSMITH_PROJECT", "pageglance"),
    )

    endpoint = os.getenv("LANGSMITH_ENDPOINT")

    if endpoint:
        os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint)

    return True


@contextmanager
def trace_run(
    name: str,
    run_type: str = "chain",
    inputs: Optional[dict] = None,
    metadata: Optional[dict] = None,
    tags: Optional[list] = None,
) -> Iterator[Any]:
    """Create an optional LangSmith trace without hiding application errors."""

    if not is_tracing_enabled():
        yield None
        return

    try:
        from langsmith.run_helpers import trace

        ctx = trace(
            name=name,
            run_type=run_type,
            inputs=inputs,
            metadata=metadata or {},
            tags=tags or [],
        )
        run = ctx.__enter__()

    except Exception:
        yield None
        return

    try:
        yield run

    except BaseException as exc:
        try:
            ctx.__exit__(type(exc), exc, exc.__traceback__)
        except Exception:
            pass
        raise

    else:
        try:
            ctx.__exit__(None, None, None)
        except Exception:
            pass


def trace_start(
    name: str,
    run_type: str = "chain",
    inputs: Optional[dict] = None,
    metadata: Optional[dict] = None,
):
    if not is_tracing_enabled():
        return None

    try:
        from langsmith.run_helpers import trace

        ctx = trace(
            name=name,
            run_type=run_type,
            inputs=inputs,
            metadata=metadata or {},
        )
        run = ctx.__enter__()

        return (ctx, run)

    except Exception:
        return None


def trace_end(
    handle: Optional[tuple],
    outputs: Optional[dict] = None,
    error: Optional[Exception] = None,
):
    if not handle:
        return

    ctx, run = handle

    try:
        if error is not None:
            ctx.__exit__(
                type(error),
                error,
                error.__traceback__,
            )
        else:
            ctx.__exit__(None, None, None)

    except Exception:
        pass


configure()
