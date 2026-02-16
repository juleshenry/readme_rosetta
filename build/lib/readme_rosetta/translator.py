import hashlib
import json
import logging
import os
import re
from typing import Dict, List, Optional

import ollama

from .lang_codes import lang_codes

logger = logging.getLogger(__name__)


class Translator:
    """Handles translation using Ollama LLM models."""

    def __init__(
        self,
        model_id: str = "llama3.2",
        local_files_only: bool = True,
        device: Optional[str] = None,
        cache_path: str = ".rosetta_cache.json",
    ) -> None:
        """
        Initialize the Translator.

        :param model_id: The identifier of the Ollama model to use.
        :param local_files_only: Whether to only use locally available files (legacy).
        :param device: The device to run the model on (legacy).
        :param cache_path: Path to the translation cache file.
        """
        self.model_id = model_id
        self.cache_path = cache_path
        self.cache: Dict[str, str] = self._load_cache()

    def _load_cache(self) -> Dict[str, str]:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
        return {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def _get_cache_key(self, text: str, from_code: str, to_code: str) -> str:
        content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"{from_code}:{to_code}:{content_hash}"

    def load(self, local_files_only: Optional[bool] = None) -> None:
        """
        Ensure the model is available in Ollama.

        :param local_files_only: Legacy parameter for compatibility.
        """
        try:
            logger.info(f"Checking Ollama model: {self.model_id}...")
            try:
                ollama.show(self.model_id)
            except Exception:
                logger.info(f"Model {self.model_id} not found, pulling...")
                ollama.pull(self.model_id)
        except Exception as e:
            logger.error(f"Error ensuring model: {e}")

    def translate(self, text: str, from_code: str, to_code: str) -> str:
        """
        Translate a single string from one language to another.

        :param text: The text to translate.
        :param from_code: The source language code.
        :param to_code: The target language code.
        :return: The translated text.
        """
        if not text.strip():
            return text

        cache_key = self._get_cache_key(text, from_code, to_code)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # For single words/short strings, use a simpler query
        if len(text.split()) < 3:
            result = self._query_ollama(text, from_code, to_code, simple=True)
        else:
            result = self.translate_batch([text], from_code, to_code)[0]

        self.cache[cache_key] = result
        self._save_cache()
        return result

    def _query_ollama(
        self, text: str, from_code: str, to_code: str, simple: bool = False
    ) -> str:
        """
        Internal method to query Ollama for translation.

        :param text: The text to translate.
        :param from_code: The source language code.
        :param to_code: The target language code.
        :param simple: Whether to use a simpler prompt for short texts.
        :return: The translated text.
        """
        from_lang = lang_codes.get(from_code, from_code)
        to_lang = lang_codes.get(to_code, to_code)

        system_msg = (
            f"You are a professional translator from {from_lang} to {to_lang}. "
            "Respond ONLY with the translated text. No explanations, no notes."
        )
        if not simple:
            system_msg += (
                " Preserve all placeholders like ROSETTA_CB_0, ROSETTA_CB_1, "
                "or ROSETTA_RST_0 exactly. Do NOT add extra placeholders or headers."
            )

        try:
            response = ollama.chat(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": text},
                ],
            )
            translated = response["message"]["content"].strip()

            # Clean up common LLM hallucinations related to placeholders
            lines = translated.splitlines()
            cleaned_lines = []
            for line in lines:
                # Skip invalid placeholders or metadata headers
                if re.search(r"ROSETTA_(CB|RST)_(?!(\d+)\b)", line, re.IGNORECASE):
                    continue
                if any(
                    x in line
                    for x in [
                        "ROSETTA_CB_SOURCE",
                        "ROSETTA_CB_TARGET",
                        "ROSETTA_CB_NEXTRA",
                    ]
                ):
                    continue
                cleaned_lines.append(line)

            translated = "\n".join(cleaned_lines)

            # Clean up common LLM artifacts
            if translated.startswith('"') and translated.endswith('"'):
                translated = translated[1:-1].strip()

            # Preserve trailing newline if it existed
            if text.endswith("\n") and not translated.endswith("\n"):
                translated += "\n"
            return translated
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return text

    def translate_batch(
        self, texts: List[str], from_code: str, to_code: str
    ) -> List[str]:
        """
        Translate a list of strings from one language to another.

        :param texts: The list of strings to translate.
        :param from_code: The source language code.
        :param to_code: The target language code.
        :return: A list of translated strings.
        """
        if not texts:
            return []

        results = []
        for text in texts:
            if not text.strip():
                results.append(text)
                continue

            cache_key = self._get_cache_key(text, from_code, to_code)
            if cache_key in self.cache:
                results.append(self.cache[cache_key])
                continue

            translated = self._query_ollama(text, from_code, to_code)
            results.append(translated)
            self.cache[cache_key] = translated

        self._save_cache()
        return results
