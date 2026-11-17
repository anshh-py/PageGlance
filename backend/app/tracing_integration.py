"""Integration helpers to trace the actual pipeline."""

from app import tracing_config
from app.tracing import trace, traceable
from app.tracing_instrumentation import TracedRagPipeline

# Ensure tracing is configured on import
tracing_config.configure_tracing()

__all__ = ["trace", "traceable", "TracedRagPipeline", "tracing_config"]
