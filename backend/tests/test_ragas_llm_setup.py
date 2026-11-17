"""Regression tests for Ragas LLM setup with Groq structured output."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_ragas_llm_uses_instructor_json_schema_mode():
    """Ensure Ragas LLM setup uses a compatible instructor mode for Groq."""
    from evals import run_ragas_eval
    
    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key", "GROQ_MODEL": "openai/gpt-oss-20b"}):
        with patch("evals.run_ragas_eval.OpenAI") as mock_openai:
            with patch("evals.run_ragas_eval.instructor") as mock_instructor:
                with patch("evals.run_ragas_eval.InstructorLLM") as mock_instructor_llm:
                    mock_client = MagicMock()
                    mock_openai.return_value = mock_client
                    mock_patched = MagicMock()
                    mock_instructor.from_openai.return_value = mock_patched
                    mock_instructor.Mode.JSON_SCHEMA = "json_schema"
                    mock_instructor.Mode.MD_JSON = "md_json"
                    mock_instructor.Mode.JSON = "json"
                    
                    mock_llm = MagicMock()
                    mock_instructor_llm.return_value = mock_llm
                    
                    llm = run_ragas_eval._setup_llm()
                    
                    # Should try JSON_SCHEMA first
                    mock_instructor.from_openai.assert_called_once()
                    call_args = mock_instructor.from_openai.call_args
                    assert call_args[0][0] == mock_client
                    # Should create InstructorLLM with patched client
                    mock_instructor_llm.assert_called_once()
                    assert call_args[1].get('mode') == mock_instructor.Mode.JSON_SCHEMA or call_args.kwargs.get('mode') == mock_instructor.Mode.JSON_SCHEMA


def test_ragas_llm_fallback_modes():
    """Test fallback to other modes if JSON_SCHEMA fails."""
    from evals import run_ragas_eval
    
    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        with patch("evals.run_ragas_eval.OpenAI") as mock_openai:
            with patch("evals.run_ragas_eval.instructor") as mock_instructor:
                with patch("evals.run_ragas_eval.InstructorLLM") as mock_instructor_llm:
                    mock_client = MagicMock()
                    mock_openai.return_value = mock_client
                    mock_instructor.Mode.JSON_SCHEMA = "json_schema"
                    mock_instructor.Mode.MD_JSON = "md_json"
                    mock_instructor.Mode.JSON = "json"
                    
                    # Make JSON_SCHEMA fail
                    mock_instructor.from_openai.side_effect = [Exception("fail"), MagicMock()]
                    mock_llm = MagicMock()
                    mock_instructor_llm.return_value = mock_llm
                    
                    llm = run_ragas_eval._setup_llm()
                    assert mock_instructor.from_openai.call_count == 2
