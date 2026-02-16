import argparse
import logging
import os
import sys
from typing import Any, Dict

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
    level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger directly to allow re-configuration
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = RichHandler(rich_tracebacks=True, console=console)
    root.addHandler(handler)

    # Set levels for noisy libraries
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
        logging.info("Detected pyproject.toml configuration.")
        try:
            with open("pyproject.toml", "rb") as f:
                pyproject = tomllib.load(f)
                config = pyproject.get("tool", {}).get("readme-rosetta", {})
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not load pyproject.toml: {e}[/yellow]"
            )
    return config


def get_git_remote_url() -> str:
    """
    Attempts to get the git remote origin URL.
    """
    import subprocess

    try:
        url = (
            subprocess.check_output(
                ["git", "remote", "get-url", "origin"], stderr=subprocess.STDOUT
            )
            .decode("utf-8")
            .strip()
        )
        if url.endswith(".git"):
            url = url[:-4]
        # Convert SSH to HTTPS for better linking
        if url.startswith("git@"):
            url = url.replace(":", "/").replace("git@", "https://")
        return url
    except Exception:
        return ""


def main() -> None:
    """
    Main entry point for the README Rosetta CLI.
    """
    # Initialize basic logging early so we can log during configuration loading
    setup_logging(False)
    config = load_config()

    parser = argparse.ArgumentParser(
        description="README Rosetta: Translate your documentation."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=config.get("path", "README.md"),
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
    parser.add_argument(
        "--base-url",
        default=config.get("base-url", ""),
        help="Base URL for absolute links (auto-detected if not provided).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        default=config.get("raw", False),
        help="Use absolute raw links with ?raw=true.",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    base_url = args.base_url
    if not base_url:
        base_url = get_git_remote_url()
        if base_url:
            # For GitHub, we want to link to the blob/main (or master)
            # This is a bit of a guess, but common
            if "github.com" in base_url and "/blob/" not in base_url:
                base_url = f"{base_url}/blob/main"

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
            if project_path == "README.md" and not os.path.exists("README.md"):
                project_path = "."

            if args.dry_run:
                console.print(
                    f"[blue]DRY RUN: Would setup Sphinx for {args.langs}[/blue]"
                )
            else:
                sphinx_handler.setup_sphinx(project_path, args.langs, args.src_lang)

        if args.langs:
            src_file = args.path
            start_code = args.src_lang
            target_codes = args.langs
            output_file = args.readme

            # If path is a directory, look for README.md inside it
            if os.path.isdir(src_file):
                src_file = os.path.join(src_file, "README.md")

            if not os.path.exists(src_file):
                if not args.sphinx:  # Only error if we're not just doing sphinx
                    console.print(f"[red]Error: {src_file} not found.[/red]")
                    sys.exit(1)
                else:
                    console.print(
                        f"[yellow]Skipping README translation: {src_file} not found.[/yellow]"
                    )
            else:
                logging.info(f"Source file detected: {src_file}")

                # Discover existing translations to populate the stone
                discovered_md_langs = md_handler.discover_translations(src_file)
                discovered_sphinx_langs = sphinx_handler.discover_translations()
                discovered_langs = sorted(
                    list(set(discovered_md_langs) | set(discovered_sphinx_langs))
                )

                if discovered_langs:
                    logging.info(
                        f"Discovered existing translations: {', '.join(discovered_langs)}"
                    )

                # We need the first header to forge the stone correctly
                with open(src_file, "r", encoding="utf-8") as f:
                    first_header = ""
                    for line in f:
                        if line.startswith("#") and "Documentation Support" not in line:
                            first_header = line
                            break

                # Initialize current_rosetta with existing table if it exists
                # translate_markdown will handle extracting it from the file during the first run
                current_rosetta = ""

                console.print(
                    f"[bold green]Starting translation of {src_file} to "
                    f"{len(target_codes)} languages...[/bold green]"
                )

                translated_sections = []
                source_without_rosetta = ""

                for code in tqdm(target_codes, desc="Languages", position=0):
                    target_file = None
                    if not args.no_split:
                        base, ext = os.path.splitext(output_file)
                        target_file = f"{base}.{code}{ext}"
                        if os.path.exists(target_file):
                            logging.info(f"Target file detected: {target_file}")

                    trans_text, rosetta_part, source_clean, identified_header = (
                        md_handler.translate_markdown(
                            src_file,
                            start_code,
                            code,
                            pbar_pos=1,
                            dry_run=args.dry_run,
                            add_lang_to_header=args.no_split,
                            target_file=target_file,
                            base_url=base_url,
                            raw=args.raw,
                        )
                    )
                    source_without_rosetta = source_clean

                    # Extract first header if not already found
                    if not first_header:
                        first_header = identified_header

                    if args.no_split:
                        translated_sections.append(trans_text)
                        if not current_rosetta:
                            current_rosetta = rosetta_part
                        else:
                            current_rosetta = md_handler.forge_stone(
                                first_header,
                                start_code,
                                code,
                                existing_table=current_rosetta,
                                base_url=base_url,
                                raw=args.raw,
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
                        if not current_rosetta:
                            current_rosetta = rosetta_part
                        else:
                            current_rosetta = md_handler.forge_stone(
                                first_header,
                                start_code,
                                code,
                                existing_table=current_rosetta,
                                target_file=lang_output,
                                base_url=base_url,
                                raw=args.raw,
                            )

                # After the translation loop, ensure ALL discovered translations are in the stone
                for code in discovered_langs:
                    if code == start_code:
                        continue

                    base, ext = os.path.splitext(output_file)
                    lang_output = f"{base}.{code}{ext}"
                    current_rosetta = md_handler.forge_stone(
                        first_header,
                        start_code,
                        code,
                        existing_table=current_rosetta,
                        target_file=lang_output,
                        base_url=base_url,
                        raw=args.raw,
                    )

                if not args.dry_run:
                    with open(output_file, "w", encoding="utf-8") as out:
                        out.write(current_rosetta)
                        out.write("\n")
                        out.write(source_without_rosetta)
                        if args.no_split:
                            for section in translated_sections:
                                out.write("\n\n<!-- <Rosetta Translations> -->\n\n")
                                out.write(section)
                console.print(f"[bold green]Done! {output_file} updated.[/bold green]")

        if args.gitbook and not args.dry_run:
            # Include both new translations and existing discovered ones
            discovered_md_langs = md_handler.discover_translations(args.readme)
            discovered_sphinx_langs = sphinx_handler.discover_translations()
            discovered_langs = sorted(
                list(set(discovered_md_langs) | set(discovered_sphinx_langs))
            )
            all_langs = sorted(list(set(args.langs) | set(discovered_langs)))

            if not all_langs:
                console.print(
                    "[yellow]Warning: --gitbook requires translations to generate links.[/yellow]"
                )
            else:
                with open("SUMMARY.md", "w", encoding="utf-8") as summary:
                    summary.write("# Summary\n\n")
                    summary.write(f"* [Introduction]({args.readme})\n")
                    for code in all_langs:
                        if code == args.src_lang:
                            continue
                        lang_name = str(lang_codes.get(code, code))
                        if args.no_split:
                            summary.write(f"* [{lang_name}]({args.readme})\n")
                        else:
                            base, ext = os.path.splitext(args.readme)
                            lang_output = f"{base}.{code}{ext}"
                            summary.write(f"* [{lang_name}]({lang_output})\n")
                console.print("[green]SUMMARY.md created for GitBook.[/green]")

        if not args.sphinx and not args.langs:
            # Default single translation (legacy)
            start_code = args.src_lang
            md_text = args.path
            end_code = "es"

            if not os.path.exists(md_text):
                console.print(f"[red]Error: {md_text} not found.[/red]")
                sys.exit(1)

            logging.info(f"Source file detected: {md_text}")
            console.print(f"Translating {md_text} from {start_code} to {end_code}...")
            translated_text, rosetta_table, source_clean, identified_header = (
                md_handler.translate_markdown(
                    md_text, start_code, end_code, dry_run=args.dry_run
                )
            )

            if not args.dry_run:
                with open(args.readme, "w", encoding="utf-8") as nrm:
                    nrm.write(rosetta_table)
                    nrm.write(source_clean)
                    nrm.write("\n<!-- <Rosetta Translations> -->\n")
                    nrm.write(translated_text)
            console.print(f"[bold green]Done! See {args.readme}[/bold green]")

    except Exception as e:
        if args.verbose:
            raise e
        console.print(f"[red]An error occurred: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
