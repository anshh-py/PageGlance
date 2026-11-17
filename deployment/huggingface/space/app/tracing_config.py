"""LangSmith configuration setup."""

import os


def configure_tracing():
    if os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"false", "0", "no", "off"}:
        return False
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        return False
    # Set env vars for langsmith SDK
    os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGSMITH_TRACING", "true"))
    os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "pageglance"))
    if os.getenv("LANGSMITH_ENDPOINT"):
        os.environ.setdefault("LANGCHAIN_ENDPOINT", os.getenv("LANGSMITH_ENDPOINT"))
    return True


configure_tracing()
