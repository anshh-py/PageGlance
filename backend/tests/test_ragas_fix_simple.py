def test_ragas_setup_uses_instructor():
    from evals import run_ragas_eval
    import inspect
    src = inspect.getsource(run_ragas_eval._setup_llm)
    assert 'instructor' in src
    assert 'JSON_SCHEMA' in src or 'MD_JSON' in src
