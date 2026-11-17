"""Tests for LangSmith tracing integration."""

import os
from unittest.mock import patch, MagicMock


def test_tracing_disabled_by_default():
    from app import tracing
    # Reset
    tracing._TRACING_ENABLED = None
    with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}, clear=False):
        tracing._TRACING_ENABLED = None
        assert tracing._is_tracing_enabled() is False


def test_tracing_enabled_with_api_key():
    from app import tracing
    tracing._TRACING_ENABLED = None
    with patch.dict(os.environ, {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "test-key"}, clear=False):
        tracing._TRACING_ENABLED = None
        assert tracing._is_tracing_enabled() is True


def test_tracing_config_configures_env():
    from app import tracing_config
    with patch.dict(os.environ, {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "test-key", "LANGSMITH_PROJECT": "test-proj"}, clear=False):
        result = tracing_config.configure_tracing()
        assert result is True
        assert os.environ.get("LANGCHAIN_TRACING_V2") is not None


def test_traceable_decorator_noop_when_disabled():
    from app import tracing
    tracing._TRACING_ENABLED = None
    with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}, clear=False):
        tracing._TRACING_ENABLED = None
        @tracing.traceable
        def foo():
            return 42
        assert foo() == 42


def test_traced_pipeline_components_exist():
    from app.tracing_instrumentation import TracedRagPipeline
    assert hasattr(TracedRagPipeline, 'trace_api_request')
    assert hasattr(TracedRagPipeline, 'trace_query_transformation')
    assert hasattr(TracedRagPipeline, 'trace_hybrid_retrieval')
    assert hasattr(TracedRagPipeline, 'trace_vector_retrieval')
    assert hasattr(TracedRagPipeline, 'trace_fts_retrieval')
    assert hasattr(TracedRagPipeline, 'trace_rrf_fusion')
    assert hasattr(TracedRagPipeline, 'trace_reranking')
    assert hasattr(TracedRagPipeline, 'trace_context_assembly')
    assert hasattr(TracedRagPipeline, 'trace_llm_generation')
    assert hasattr(TracedRagPipeline, 'trace_citation_validation')
