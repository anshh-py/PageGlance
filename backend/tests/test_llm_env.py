"""Test LLM env loading."""
from app import llm


def test_get_answer_service_caches():
    llm.get_answer_service.cache_clear()
    from app.llm import get_answer_service
    s1 = get_answer_service()
    s2 = get_answer_service()
    assert s1 is s2
