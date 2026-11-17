'''Regression tests for Ragas Groq JSON validation fix.'''

import os
from unittest.mock import MagicMock, patch


def test_ragas_setup_uses_instructor_with_json_schema():
    from evals import run_ragas_eval
    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key", "GROQ_MODEL": "openai/gpt-oss-20b"}):
        with patch("evals.run_ragas_eval.OpenAI") as mock_openai:
            with patch("evals.run_ragas_eval.instructor") as mock_instructor:
                with patch("evals.run_ragas_eval.InstructorLLM") as mock_llm_cls:
                    mock_client = MagicMock()
                    mock_openai.return_value = mock_client
                    mock_patched = MagicMock()
                    mock_instructor.from_openai.return_value = mock_patched
                    mock_instructor.Mode.JSON_SCHEMA = "JSON_SCHEMA"
                    mock_instructor.Mode.MD_JSON = "MD_JSON"
                    mock_instructor.Mode.JSON = "JSON"
                    mock_llm_cls.return_value = MagicMock()
                    
                    llm = run_ragas_eval._setup_llm()
                    
                    mock_instructor.from_openai.assert_called_once()
                    args, kwargs = mock_instructor.from_openai.call_args
                    assert args[0] == mock_client
                    assert kwargs.get("mode") == mock_instructor.Mode.JSON_SCHEMA
                    mock_llm_cls.assert_called_once()
