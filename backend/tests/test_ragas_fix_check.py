'''Check fix is applied.'''
import inspect
from evals import run_ragas_eval


def test_fix_applied():
    src = inspect.getsource(run_ragas_eval._setup_llm)
    assert 'instructor' in src
    assert 'JSON_SCHEMA' in src
    assert 'InstructorLLM' in src
