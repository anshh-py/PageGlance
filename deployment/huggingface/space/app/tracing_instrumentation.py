"""Comprehensive tracing instrumentation for the RAG pipeline."""

import time
from app.tracing import trace


class TracedRagPipeline:
    @staticmethod
    def trace_api_request(func, *args, **kwargs):
        with trace(name="api.request", run_type="chain", endpoint="ask"):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                raise

    @staticmethod
    def trace_query_transformation(func, *args, **kwargs):
        with trace(name="query.transformation", run_type="chain"):
            return func(*args, **kwargs)

    @staticmethod
    def trace_hybrid_retrieval(func, *args, **kwargs):
        with trace(name="retrieval.hybrid", run_type="retriever"):
            return func(*args, **kwargs)

    @staticmethod
    def trace_vector_retrieval(func, *args, **kwargs):
        with trace(name="retrieval.vector", run_type="retriever"):
            return func(*args, **kwargs)

    @staticmethod
    def trace_fts_retrieval(func, *args, **kwargs):
        with trace(name="retrieval.fts", run_type="retriever"):
            return func(*args, **kwargs)

    @staticmethod
    def trace_rrf_fusion(func, *args, **kwargs):
        with trace(name="retrieval.rrf_fusion", run_type="chain"):
            return func(*args, **kwargs)

    @staticmethod
    def trace_reranking(func, *args, **kwargs):
        with trace(name="retrieval.reranking", run_type="chain"):
            return func(*args, **kwargs)

    @staticmethod
    def trace_context_assembly(func, *args, **kwargs):
        with trace(name="context.assembly", run_type="chain"):
            return func(*args, **kwargs)

    @staticmethod
    def trace_llm_generation(func, *args, **kwargs):
        with trace(name="llm.generation", run_type="llm"):
            return func(*args, **kwargs)

    @staticmethod
    def trace_citation_validation(func, *args, **kwargs):
        with trace(name="answer.citation_validation", run_type="chain"):
            return func(*args, **kwargs)
