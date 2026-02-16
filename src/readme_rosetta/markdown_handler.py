import logging
import os
import re
from typing import List, Optional, Tuple

from .lang_codes import lang_codes
from .translator import Translator

logger = logging.getLogger(__name__)


class MarkdownHandler:
    """Handles parsing and translation of Markdown files."""

    def __init__(self, translator: Translator) -> None:
        """
        Initialize the MarkdownHandler.

        :param translator: The Translator instance to use.
        """
        self.translator = translator

    def protect_code_blocks(self, text: str) -> Tuple[str, List[str]]:
        """
        Replaces code blocks with placeholders to prevent translation.

        :param text: The Markdown text.
        :return: A tuple containing the protected text and original code blocks.
        """
        placeholders = []

        def replace(match):
            placeholder = f"ROSETTA_CB_{len(placeholders)}"
            placeholders.append(match.group(0))
            return placeholder

        # Protect triple backtick blocks
        protected_text = re.sub(r"```[\s\S]*?```", replace, text)
        # Protect inline code
        protected_text = re.sub(r"`[^`\n]+`", replace, protected_text)

        return protected_text, placeholders

    def restore_code_blocks(self, text: str, placeholders: List[str]) -> str:
        """
        Restores code blocks from placeholders after translation.

        :param text: The translated text with placeholders.
        :param placeholders: The list of original code blocks.
        :return: The restored Markdown text.
        """

        def replace_match(match):
            try:
                index = int(match.group(1))
                if index < len(placeholders):
                    return placeholders[index]
            except Exception:
                pass
            return match.group(0)

        # Regex to find ROSETTA_CB_N with potential minor alterations by the model
        pattern = re.compile(r"ROSETTA[_\s-]*CB[_\s-]*(\d+)", re.IGNORECASE)
        return pattern.sub(replace_match, text)

    def clean_header_for_link(self, header: str) -> str:
        """
        Cleans a Markdown header to be used as an anchor link.

        :param header: The header text.
        :return: The cleaned header link (e.g., "#my-header").
        """
        header = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", header)
        header = header.strip().lstrip("#").strip()
        return "#" + header.lower().replace(" ", "-")

    def forge_stone(
        self,
        first_header: str,
        start_code: str,
        end_code: str,
        existing_table: Optional[str] = None,
        target_file: Optional[str] = None,
        base_url: Optional[str] = None,
        raw: bool = False,
    ) -> str:
        """
        Generates or updates the Rosetta translation table in the README.

        :param first_header: The first header of the document.
        :param start_code: The source language code.
        :param end_code: The target language code.
        :param existing_table: The existing Rosetta table, if any.
        :param target_file: The name of the target file (for split mode).
        :param base_url: The base URL for absolute links.
        :param raw: Whether to append ?raw=true to links.
        :return: The updated Rosetta table Markdown.
        """
        header_link = self.clean_header_for_link(first_header)
        lang_name = str(lang_codes.get(end_code, end_code))

        translated_lang_name = (
            self.translator.translate(lang_name, start_code, end_code)
            .strip()
            .split("\n")[0]
            .strip()
        )

        # Clean the translated name for the link
        clean_lang_link = translated_lang_name.lower().replace(" ", "-")
        # Remove common punctuation that might be added
        clean_lang_link = re.sub(r"[^\w-]", "", clean_lang_link)

        if target_file:
            if base_url:
                target_url = f"{base_url.rstrip('/')}/{target_file}"
                if raw:
                    target_url += "?raw=true"
                new_header_link = target_url + header_link
            else:
                new_header_link = target_file + header_link
        else:
            new_header_link = header_link + "-" + clean_lang_link

        new_row = f"| {lang_name} | [Link to Head of Docs]({new_header_link}) |"

        if existing_table and "<!-- <Original README.md> -->" in existing_table:
            if f"| {lang_name} |" in existing_table:
                logger.info(f"Language {lang_name} already detected in Rosetta table.")
                return existing_table

            lines = existing_table.strip().split("\n")
            last_row_idx = -1
            for i, line in enumerate(lines):
                if line.startswith("|"):
                    last_row_idx = i

            if last_row_idx != -1:
                lines.insert(last_row_idx + 1, new_row)
                return "\n".join(lines) + "\n"

        project_link = base_url if base_url else "#"
        return f"""<!-- <Original README.md> -->
# [Documentation Support in Multiple Languages]({project_link})
| About | |
| ------ | ---- |
| English | [Link to Head of Docs]({header_link}) |
{new_row}
"""

    def translate_markdown(
        self,
        md_text_path: str,
        start_code: str,
        end_code: str,
        pbar_pos: int = 0,
        dry_run: bool = False,
        add_lang_to_header: bool = True,
        target_file: Optional[str] = None,
        base_url: Optional[str] = None,
        raw: bool = False,
    ) -> Tuple[str, str, str, str]:
        """
        Translates a Markdown file using block-level context.

        :param md_text_path: The path to the Markdown file.
        :param start_code: The source language code.
        :param end_code: The target language code.
        :param pbar_pos: The position of the progress bar.
        :param dry_run: Whether to simulate translation.
        :param add_lang_to_header: Whether to add the target language name to the first header.
        :param target_file: The name of the target file (for split mode).
        :param base_url: The base URL for absolute links.
        :param raw: Whether to append ?raw=true to links.
        :return: A tuple containing (translated_content, rosetta_table,
                 original_content_without_rosetta, identified_first_header).
        """
        if not os.path.exists(md_text_path):
            return "", "", "", ""

        from tqdm import tqdm

        with open(md_text_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        first_header = ""
        existing_table = ""
        in_rosetta = False
        content_lines = []
        original_content_without_rosetta = []

        # Extract headers and existing table
        for o in lines:
            if "<!-- <Original README.md> -->" in o:
                in_rosetta = True
                existing_table += o
                continue

            if in_rosetta:
                if o.startswith("#") and "Documentation Support" not in o:
                    in_rosetta = False
                else:
                    existing_table += o
                    continue

            if "<!-- <Rosetta Translations> -->" in o:
                break

            original_content_without_rosetta.append(o)

            if o.startswith("#") and not first_header:
                first_header = o
                if add_lang_to_header:
                    lang_name = str(lang_codes.get(end_code, end_code))
                    header_for_translation = o.strip() + " (" + lang_name + ")\n"
                    content_lines.append(header_for_translation)
                else:
                    content_lines.append(o)
            else:
                content_lines.append(o)

        full_text = "".join(content_lines)
        source_without_rosetta = "".join(original_content_without_rosetta)

        if dry_run:
            return (
                "[DRY RUN] Translated Content",
                "[DRY RUN] Rosetta Table",
                source_without_rosetta,
                first_header,
            )

        # Protect code blocks
        protected_text, placeholders = self.protect_code_blocks(full_text)

        # For small enough files, translate the whole thing at once for better context
        # Otherwise, split by sections (headers)
        if len(protected_text) < 4000:
            logger.info("File size < 4000 chars, translating as a single chunk for better context.")
            final_translated_text = self.translator.translate(
                protected_text, start_code, end_code
            )
        else:
            # Split by headers (keeping the header with the following block)
            # We use a lookahead to split BEFORE headers
            parts = re.split(r"(?m)^(?=#+ )", protected_text)
            translated_parts = []
            
            logger.info(f"File size >= 4000 chars, split into {len(parts)} sections for translation.")

            pbar = tqdm(
                total=len([p for p in parts if p.strip()]),
                desc=f"Translating to {end_code}",
                leave=False,
                position=pbar_pos,
            )

            for part in parts:
                if not part.strip():
                    translated_parts.append(part)
                    continue

                translated_parts.append(
                    self.translator.translate(part, start_code, end_code)
                )
                pbar.update(1)

            pbar.close()
            final_translated_text = "".join(translated_parts)

        # Restore code blocks
        final_text = self.restore_code_blocks(final_translated_text, placeholders)

        rosetta_table = self.forge_stone(
            first_header,
            start_code,
            end_code,
            existing_table,
            target_file=target_file,
            base_url=base_url,
            raw=raw,
        )

        return final_text, rosetta_table, source_without_rosetta, first_header
