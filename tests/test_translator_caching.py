import json
import os
from unittest.mock import MagicMock, patch

import pytest
from readme_rosetta.translator import Translator


@pytest.fixture
def temp_cache(tmp_path):
    cache_file = tmp_path / ".rosetta_cache.json"
    return str(cache_file)


def test_translator_caching(temp_cache):
    # Mock ollama.chat
    with patch("ollama.chat") as mock_chat:
        mock_chat.return_value = {"message": {"content": "Hola Mundo"}}

        translator = Translator(cache_path=temp_cache)

        # First call should hit the "API"
        res1 = translator.translate("Hello World", "en", "es")
        assert res1 == "Hola Mundo"
        assert mock_chat.call_count == 1

        # Second call should hit the cache
        res2 = translator.translate("Hello World", "en", "es")
        assert res2 == "Hola Mundo"
        assert mock_chat.call_count == 1  # Still 1

        # Check cache file exists
        assert os.path.exists(temp_cache)
        with open(temp_cache, "r") as f:
            cache_data = json.load(f)
            assert len(cache_data) == 1


def test_translator_batch_caching(temp_cache):
    with patch("ollama.chat") as mock_chat:
        mock_chat.return_value = {"message": {"content": "Traducido"}}

        translator = Translator(cache_path=temp_cache)

        # Mix of cached and non-cached
        translator.cache[translator._get_cache_key("Cached", "en", "es")] = "Ya existia"

        results = translator.translate_batch(["Cached", "New"], "en", "es")
        assert results == ["Ya existia", "Traducido"]
        assert mock_chat.call_count == 1
