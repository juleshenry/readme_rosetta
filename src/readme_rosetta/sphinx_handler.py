"""
Sphinx i18n: gettext catalogs translated with the same segment translator.
"""

import logging
import os
import re
import subprocess
import sys
from typing import List

try:
    import polib
except ImportError:
    polib = None
from rich.console import Console

from .translator import Translator

logger = logging.getLogger(__name__)
console = Console()


class SphinxHandler:
    """Handles Sphinx documentation translation using gettext and PO files."""

    def __init__(self, translator: Translator) -> None:
        """
        Initialize the SphinxHandler.

        :param translator: The Translator instance to use.
        """
        self.translator = translator

    def fix_rst_underlines(self, text: str) -> str:
        """
        Ensures rST underlines (===, ---, etc.) are at least as long as the preceding text.

        :param text: The rST text.
        :return: The rST text with fixed underlines.
        """
        lines = text.splitlines()
        fixed_lines = []
        for i in range(len(lines)):
            line = lines[i]
            if i > 0 and re.match(r"^([=~`'\-\^\"*+#])\1+$", line):
                prev_line = lines[i - 1].strip()
                if prev_line:
                    char = line[0]
                    if len(line) < len(prev_line):
                        line = char * len(prev_line)
            fixed_lines.append(line)
        return "\n".join(fixed_lines)

    def discover_translations(self) -> List[str]:
        """
        Scans for existing Sphinx translations in docs/source/locale.

        :return: List of language codes found.
        """
        locale_dir = os.path.join(os.getcwd(), "docs", "source", "locale")
        if not os.path.exists(locale_dir):
            return []

        langs = []
        for d in os.listdir(locale_dir):
            if os.path.isdir(os.path.join(locale_dir, d)):
                if os.path.exists(os.path.join(locale_dir, d, "LC_MESSAGES")):
                    langs.append(d)
        return sorted(langs)

    def translate_po_file(self, po_path: str, to_code: str) -> bool:
        """
        Translates a Sphinx PO file.

        :param po_path: The path to the .po file.
        :param to_code: The target language code.
        :return: True if any translations were made, False otherwise.
        """
        if not polib:
            logger.warning("polib not installed, skipping PO translation")
            return False

        po = polib.pofile(po_path)
        entries_to_translate = [
            e for e in po if e.msgid and (not e.msgstr or "fuzzy" in e.flags)
        ]

        if not entries_to_translate:
            return False

        console.print(
            f"Translating {len(entries_to_translate)} entries in "
            f"{os.path.basename(po_path)} to {to_code}..."
        )
        msgids = [e.msgid for e in entries_to_translate]
        translated = self.translator.translate_many(msgids, to_code, syntax="rst")

        for entry, trans in zip(entries_to_translate, translated):
            if trans == entry.msgid and self.translator.needs_translation(
                entry.msgid, "rst"
            ):
                # Translation failed validation; leave it empty so Sphinx shows the source.
                continue
            entry.msgstr = self.fix_rst_underlines(trans)
            if "fuzzy" in entry.flags:
                entry.flags.remove("fuzzy")

        po.save()
        return True

    def setup_sphinx(
        self, project_path: str, langs: List[str], start_code: str = "en"
    ) -> None:
        """
        Sets up Sphinx documentation, generates API docs, and translates them.

        :param project_path: The path to the project source code.
        :param langs: A list of target language codes.
        :param start_code: The source language code.
        """
        docs_dir = os.path.join(os.getcwd(), "docs")
        source_dir = os.path.join(docs_dir, "source")

        python_exe = sys.executable
        project = os.path.basename(os.path.abspath(os.getcwd()))
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath(os.path.join(os.getcwd(), "src"))

        if os.path.exists(docs_dir):
            logging.info(f"Sphinx documentation directory detected: {docs_dir}")
        else:
            os.makedirs(docs_dir)
            subprocess.run(
                [
                    python_exe,
                    "-m",
                    "sphinx.cmd.quickstart",
                    "-q",
                    "--sep",
                    "-p",
                    project,
                    "-a",
                    "Author",
                    "-v",
                    "0.1.0",
                    docs_dir,
                ],
                env=env,
                check=True,
            )

        # Ensure conf.py has i18n settings and sys.path for autodoc
        conf_path = os.path.join(source_dir, "conf.py")
        if os.path.exists(conf_path):
            logging.info(f"Sphinx configuration detected: {conf_path}")
            with open(conf_path, "r", encoding="utf-8") as f:
                conf_content = f.read()

            updates = []
            if 'sys.path.insert(0, os.path.abspath("../../src"))' not in conf_content:
                updates.append(
                    'import sys, os\nsys.path.insert(0, os.path.abspath("../../src"))'
                )
            if 'locale_dirs = ["locale/"]' not in conf_content:
                updates.append('locale_dirs = ["locale/"]')
            if "gettext_compact = False" not in conf_content:
                updates.append("gettext_compact = False")

            if updates:
                with open(conf_path, "a", encoding="utf-8") as f:
                    f.write("\n" + "\n".join(updates) + "\n")

        # Run apidoc
        subprocess.run(
            [
                python_exe,
                "-m",
                "sphinx.ext.apidoc",
                "-o",
                source_dir,
                project_path,
                "-f",
            ],
            env=env,
            check=True,
        )

        # Generate gettext
        subprocess.run(
            [
                python_exe,
                "-m",
                "sphinx.cmd.build",
                "-M",
                "gettext",
                source_dir,
                os.path.join(docs_dir, "build"),
            ],
            env=env,
            check=True,
        )

        # Update and translate PO files
        for lang in langs:
            if lang == ".":
                continue

            # Check if we can skip updating if it's already done
            locale_dir = os.path.join(source_dir, "locale", lang, "LC_MESSAGES")
            html_output = os.path.join(docs_dir, "build", "html", lang)

            console.print(f"Updating catalogs for {lang}...")
            result = subprocess.run(
                [
                    python_exe,
                    "-m",
                    "sphinx_intl",
                    "update",
                    "-p",
                    os.path.join(docs_dir, "build", "gettext"),
                    "-l",
                    lang,
                ],
                cwd=docs_dir,
                env=env,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                console.print(
                    f"[red]sphinx-intl failed for {lang}:[/red] {result.stderr.strip()}"
                )
                continue

            any_translated = False
            if os.path.exists(locale_dir):
                po_files = sorted(
                    os.path.join(root, f)
                    for root, _, files in os.walk(locale_dir)
                    for f in files
                    if f.endswith(".po")
                )
                for po_path in po_files:
                    if self.translate_po_file(po_path, lang):
                        any_translated = True

            # Build HTML for this language if needed
            if not any_translated and os.path.exists(html_output):
                console.print(
                    f"[blue]Sphinx translation for {lang} is up to date. Skipping build.[/blue]"
                )
                continue

            console.print(f"Building HTML for {lang}...")
            subprocess.run(
                [
                    python_exe,
                    "-m",
                    "sphinx.cmd.build",
                    "-b",
                    "html",
                    "-D",
                    f"language={lang}",
                    source_dir,
                    os.path.join(docs_dir, "build", "html", lang),
                ],
                env=env,
            )

        console.print("[green]Sphinx setup and translation complete.[/green]")
