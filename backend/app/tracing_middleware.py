"""Tracing middleware and hooks for FastAPI."""

from app import tracing_config

tracing_config.configure_tracing()
