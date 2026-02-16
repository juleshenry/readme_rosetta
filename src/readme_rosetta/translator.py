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
                logger.info(f"Cache file detected: {self.cache_path}")
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
            logger.debug(f"Cache hit for {from_code}->{to_code}")
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

        if simple:
            system_msg = (
                f"Translate the following single word or short phrase from {from_lang} to {to_lang}.\n"
                "Respond ONLY with the translation. No punctuation unless part of the translation. No explanations."
            )
        else:
            system_msg = (
                f"You are a professional translator.\n"
                f"Source Language: {from_lang}\n"
                f"Target Language: {to_lang}\n\n"
                "RULES:\n"
                f"1. Translate the user's text precisely from {from_lang} to {to_lang}.\n"
                "2. Respond ONLY with the translated text. DO NOT add any explanations, notes, or introductions.\n"
                "3. Preserve all Markdown/rST formatting (headers, lists, bold, etc.) exactly. DO NOT add new headers or structure.\n"
                "4. DO NOT hallucinate. DO NOT add any new features, examples, sections, or information.\n"
                "5. Maintain the original structure of the text exactly. If the input is a single sentence, the output MUST be a single sentence.\n"
                "6. Do NOT add any links or code blocks that are not present in the source text.\n"
                "7. If you cannot translate something, return the original text as is.\n"
                "8. DO NOT assume the project's technology. DO NOT add 'npm install' or similar commands if they are not in the source."
            )
            found_placeholders_types = []
            if "ROSETTA_CB_" in text:
                found_placeholders_types.append("ROSETTA_CB_N")
            if "ROSETTA_RST_" in text:
                found_placeholders_types.append("ROSETTA_RST_N")

            if found_placeholders_types:
                placeholders_str = " and ".join(found_placeholders_types)
                system_msg += (
                    f"\n9. Preserve all placeholders like {placeholders_str} exactly."
                )

        # Count placeholders in source
        source_cb_count = len(re.findall(r"ROSETTA_CB_\d+", text))
        source_rst_count = len(re.findall(r"ROSETTA_RST_\d+", text))
        source_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text)
        source_urls = [l[1] for l in source_links]

        # Retry logic for common LLM failure modes
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                logger.info(
                    f"Querying {self.model_id} for translation ({from_code}->{to_code}, attempt {attempt + 1})"
                )
                logger.debug(f"Input text chunk: {text[:100]}...")

                response = ollama.chat(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": text},
                    ],
                )
                translated = response["message"]["content"].strip()
                logger.debug(f"Raw response from model: {translated[:100]}...")

                # Check if it looks like a conversational response instead of a translation
                is_bad = False
                bad_indicators = [
                    "nothing to translate",
                    "puedo ayudarte",
                    "aquí tienes la traducción",
                    "claro, aquí tienes",
                    "i am an ai",
                    "as an ai",
                    "soy un modelo",
                    "no hay nada que",
                    "here is the translation",
                ]
                if any(ind in translated.lower() for ind in bad_indicators):
                    if (
                        len(translated.split()) < len(text.split()) * 0.5
                        or len(translated.split()) < 10
                    ):
                        is_bad = True

                # Check for placeholder mismatch
                trans_cb_count = len(re.findall(r"ROSETTA_CB_\d+", translated))
                trans_rst_count = len(re.findall(r"ROSETTA_RST_\d+", translated))

                if (
                    trans_cb_count != source_cb_count
                    or trans_rst_count != source_rst_count
                ):
                    logger.warning(
                        f"Placeholder count mismatch: CB {trans_cb_count}/{source_cb_count}, RST {trans_rst_count}/{source_rst_count}"
                    )
                    is_bad = True

                # Check for hallucinated links
                trans_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", translated)
                for _, url in trans_links:
                    if url not in source_urls and not url.startswith("#"):
                        # If it's a completely new external link, it's likely a hallucination
                        if "http" in url and "github.com" in url:
                            logger.warning(f"Detected hallucinated link: {url}")
                            is_bad = True
                            break

                if is_bad and attempt < max_retries:
                    logger.warning(
                        f"Detected bad translation, retrying... (Attempt {attempt + 1})"
                    )
                    system_msg += "\nCRITICAL: DO NOT TALK. ONLY TRANSLATE. PRESERVE PLACEHOLDERS. DO NOT ADD LINKS."
                    continue

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

                # Cross-check: If translation is empty but source wasn't, or it's mostly garbage
                if not translated.strip() and text.strip():
                    if attempt < max_retries:
                        continue
                    return text  # Fallback to original

                return translated
            except Exception as e:
                logger.error(f"Ollama error (attempt {attempt}): {e}")
                if attempt == max_retries:
                    return text
        return text  # Ultimate fallback

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

        from tqdm import tqdm

        results = []
        # Use progress bar if there's more than one item
        iterable = (
            tqdm(texts, desc="Translating items", leave=False)
            if len(texts) > 1
            else texts
        )

        for text in iterable:
            if not text.strip():
                results.append(text)
                continue

            cache_key = self._get_cache_key(text, from_code, to_code)
            if cache_key in self.cache:
                logger.debug(f"Cache hit for {from_code}->{to_code}")
                results.append(self.cache[cache_key])
                continue

            translated = self._query_ollama(text, from_code, to_code)
            results.append(translated)
            self.cache[cache_key] = translated

        self._save_cache()
        return results
