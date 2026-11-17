'''Regression test for Groq JSON validation fix.'''
import inspect
from evals import run_ragas_eval


def test_ragas_llm_uses_instructor_json_schema():
    src = inspect.getsource(run_ragas_eval._setup_llm)
    assert 'instructor' in src
    assert 'JSON_SCHEMA' in src
    assert 'InstructorLLM' in src
