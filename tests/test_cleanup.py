import pytest
from unittest.mock import MagicMock, patch
from readme_rosetta.translator import Translator


@patch("ollama.chat")
def test_translator_hallucination_cleanup(mock_chat):
    # Mocking the response from Ollama with hallucinations
    mock_chat.return_value = {
        "message": {
            "content": """Translated text.
ROSETTA_CB_N
ROSETTA_CB_N: true
ROSETTA_CB_SOURCE: English
ROSETTA_CB_TARGET: German"""
        }
    }

    translator = Translator()
    result = translator.translate("Some text", "en", "de")

    assert "ROSETTA_CB_SOURCE" not in result
    assert "ROSETTA_CB_N: true" not in result
    assert "ROSETTA_CB_N" not in result
    assert result.strip() == "Translated text."


@patch("ollama.chat")
def test_translator_preserves_legit_placeholders(mock_chat):
    mock_chat.return_value = {
        "message": {"content": "Translated text with ROSETTA_CB_0 and ROSETTA_CB_1."}
    }

    translator = Translator()
    result = translator.translate(
        "Original text with ROSETTA_CB_0 and ROSETTA_CB_1.", "en", "de"
    )

    assert "ROSETTA_CB_0" in result
    assert "ROSETTA_CB_1" in result
    assert "ROSETTA_CB_SOURCE" not in result
