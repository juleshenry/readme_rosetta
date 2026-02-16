import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import tomllib
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm import tqdm

from .lang_codes import lang_codes
from .markdown_handler import MarkdownHandler
from .sphinx_handler import SphinxHandler
from .translator import Translator

console = Console()


def setup_logging(verbose: bool) -> None:
    """
    Sets up the logging configuration using Rich.

    :param verbose: Whether to enable verbose (DEBUG) logging.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )
    if not verbose:
        logging.getLogger("ollama").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)


def load_config() -> Dict[str, Any]:
    """
    Loads configuration from pyproject.toml if available.
    """
    config = {}
    if os.path.exists("pyproject.toml"):
        try:
            with open("pyproject.toml", "rb") as f:
                pyproject = tomllib.load(f)
                config = pyproject.get("tool", {}).get("readme-rosetta", {})
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not load pyproject.toml: {e}[/yellow]"
            )
    return config


def main() -> None:
    """
    Main entry point for the README Rosetta CLI.
    """
    config = load_config()

    parser = argparse.ArgumentParser(
        description="README Rosetta: Translate your documentation."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=config.get("path", "README_SOURCE.md"),
        help="Path to the source file or project directory.",
    )
    parser.add_argument(
        "--sphinx",
        action="store_true",
        default=config.get("sphinx", False),
        help="Setup Sphinx documentation and translate it.",
    )
    parser.add_argument(
        "--gitbook",
        action="store_true",
        default=config.get("gitbook", False),
        help="Generate GitBook compatible documentation (SUMMARY.md).",
    )
    parser.add_argument(
        "--readme",
        default=config.get("readme", "README.md"),
        help="Path to the main output README file (default: README.md).",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=config.get("langs", []),
        help="Target languages for translation.",
    )
    parser.add_argument(
        "--src-lang",
        default=config.get("src-lang", "en"),
        help="Source language code (default: en).",
    )
    parser.add_argument(
        "--model",
        default=config.get("model", "llama3.2"),
        help="Model ID to use for translation (Ollama).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate translation process."
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Append all translations to a single file instead of splitting.",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    translator = Translator(model_id=args.model)

    if not args.dry_run:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(description=f"Loading model {args.model}...", total=None)
            translator.load()

    md_handler = MarkdownHandler(translator)
    sphinx_handler = SphinxHandler(translator)

    try:
        if args.sphinx:
            if not args.langs:
                console.print("[red]Error: --langs is required with --sphinx[/red]")
                sys.exit(1)

            project_path = args.path
            if project_path == "README_SOURCE.md" and not os.path.exists(
                "README_SOURCE.md"
            ):
                project_path = "."

            if args.dry_run:
                console.print(
                    f"[blue]DRY RUN: Would setup Sphinx for {args.langs}[/blue]"
                )
            else:
                sphinx_handler.setup_sphinx(project_path, args.langs, args.src_lang)

        elif args.langs:
            src_file = args.path
            start_code = args.src_lang
            target_codes = args.langs
            output_file = args.readme

            if not os.path.exists(src_file):
                console.print(f"[red]Error: {src_file} not found.[/red]")
                sys.exit(1)

            console.print(
                f"[bold green]Starting translation of {src_file} to {len(target_codes)} languages...[/bold green]"
            )

            current_rosetta = ""
            translated_sections = []
            source_without_rosetta = ""

            for code in tqdm(target_codes, desc="Languages", position=0):
                trans_text, rosetta_part, source_clean = md_handler.translate_markdown(
                    src_file, start_code, code, pbar_pos=1, dry_run=args.dry_run
                )
                source_without_rosetta = source_clean

                if args.no_split:
                    translated_sections.append(trans_text)
                    if not current_rosetta:
                        current_rosetta = rosetta_part
                    else:
                        current_rosetta = md_handler.forge_stone(
                            "", start_code, code, current_rosetta
                        )
                else:
                    # Split mode: save to README.<lang>.md
                    base, ext = os.path.splitext(output_file)
                    lang_output = f"{base}.{code}{ext}"
                    if args.dry_run:
                        console.print(
                            f"[blue]DRY RUN: Would write to {lang_output}[/blue]"
                        )
                    else:
                        with open(lang_output, "w", encoding="utf-8") as f:
                            f.write(trans_text)

                    # Update rosetta table for main file
                    current_rosetta = md_handler.forge_stone(
                        "", start_code, code, current_rosetta
                    )

            if not args.dry_run:
                with open(output_file, "w", encoding="utf-8") as out:
                    out.write(current_rosetta)
                    out.write("\n")
                    out.write(source_without_rosetta)
                    if args.no_split:
                        for section in translated_sections:
                            out.write("\n\n<!-- toc -->\n\n")
                            out.write(section)

            if args.gitbook and not args.dry_run:
                with open("SUMMARY.md", "w", encoding="utf-8") as summary:
                    summary.write("# Summary\n\n")
                    summary.write(f"* [Introduction]({args.readme})\n")
                    for code in target_codes:
                        lang_name = str(lang_codes.get(code, code))
                        summary.write(f"* [{lang_name}]({args.readme})\n")
                console.print("[green]SUMMARY.md created for GitBook.[/green]")

            console.print(f"[bold green]Done! {output_file} updated.[/bold green]")

        else:
            # Default single translation (legacy)
            start_code = args.src_lang
            md_text = args.path
            end_code = "es"

            if not os.path.exists(md_text):
                console.print(f"[red]Error: {md_text} not found.[/red]")
                sys.exit(1)

            console.print(f"Translating {md_text} from {start_code} to {end_code}...")
            translated_text, rosetta_table, source_clean = (
                md_handler.translate_markdown(
                    md_text, start_code, end_code, dry_run=args.dry_run
                )
            )

            if not args.dry_run:
                with open(args.readme, "w", encoding="utf-8") as nrm:
                    nrm.write(rosetta_table)
                    nrm.write(source_clean)
                    nrm.write("\n<!-- toc -->\n")
                    nrm.write(translated_text)
            console.print(f"[bold green]Done! See {args.readme}[/bold green]")

    except Exception as e:
        if args.verbose:
            raise e
        console.print(f"[red]An error occurred: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
