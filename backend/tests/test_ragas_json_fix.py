"""Regression tests for Ragas JSON validation fix."""
from unittest.mock import MagicMock, patch
import os


def test_instructor_json_schema_mode_preferred():
    from evals import run_ragas_eval
    with patch.dict(os.environ, {"GROQ_API_KEY": "test", "GROQ_MODEL": "openai/gpt-oss-20b"}):
        with patch("evals.run_ragas_eval.OpenAI"):
            with patch.object(run_ragas_eval, "instructor") as mock_instructor:
                with patch.object(run_ragas_eval, "InstructorLLM") as mock_llm_cls:
                    mock_client = MagicMock()
                    run_ragas_eval.OpenAI.return_value = mock_client
                    mock_patched = MagicMock()
                    mock_instructor.from_openai.return_value = mock_patched
                    mock_instructor.Mode.JSON_SCHEMA = "json_schema"
                    mock_instructor.Mode.MD_JSON = "md_json"
                    mock_instructor.Mode.JSON = "json"
                    mock_llm_cls.return_value = MagicMock()
                    run_ragas_eval._setup_llm()
                    assert mock_instructor.from_openai.call_args[1].get("mode") == "json_schema" or mock_instructor.from_openai.call_args.kwargs.get("mode") == "json_schema"
