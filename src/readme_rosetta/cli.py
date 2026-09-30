"""
Command-line interface for README Rosetta.
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.table import Table

from . import __version__
from .backends import DEFAULT_MODELS, Backend, BackendError, create_backend
from .cache import Cache
from .lang_codes import LANGUAGES, get_language, normalize_code
from .markdown_handler import MarkdownHandler
from .readme import (
    GENERATED_NOTE,
    build_nav,
    build_unified,
    discover_translations,
    strip_rosetta,
    translation_path,
    with_nav,
)
from .slug import heading_text
from .translator import Translator

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

console = Console(stderr=True)
logger = logging.getLogger("readme_rosetta")


def setup_logging(verbose: bool) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    root.addHandler(
        RichHandler(console=console, show_path=verbose, rich_tracebacks=True)
    )
    if not verbose:
        for noisy in ("ollama", "httpx", "httpcore", "anthropic", "httpx2"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def load_config(path: str = "pyproject.toml") -> Dict[str, Any]:
    """Reads ``[tool.readme-rosetta]`` from pyproject.toml, if present."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f).get("tool", {}).get("readme-rosetta", {})
    except (OSError, tomllib.TOMLDecodeError) as e:
        console.print(f"[yellow]Warning: could not read {path}: {e}[/yellow]")
        return {}


def build_parser(config: Dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="readme-rosetta",
        description="Translate your README (and Sphinx docs) into other languages with an LLM.",
    )
    p.add_argument(
        "path",
        nargs="?",
        default=config.get("path", "README.md"),
        help="Source Markdown file, or a directory containing README.md.",
    )
    p.add_argument(
        "--langs",
        nargs="+",
        default=config.get("langs", []),
        help="Target language codes, e.g. es fr ja pt-BR.",
    )
    p.add_argument(
        "--src-lang",
        default=config.get("src-lang", "en"),
        help="Source language code (default: en).",
    )
    p.add_argument(
        "--backend",
        choices=sorted(DEFAULT_MODELS),
        default=config.get("backend", "ollama"),
        help="Translation backend (default: ollama).",
    )
    p.add_argument(
        "--model",
        default=config.get("model"),
        help=f"Model name (defaults: {', '.join(f'{k}={v}' for k, v in DEFAULT_MODELS.items())}).",
    )
    p.add_argument(
        "--ollama-host",
        default=config.get("ollama-host"),
        help="Ollama server URL (default: $OLLAMA_HOST or localhost).",
    )
    p.add_argument(
        "--readme",
        default=config.get("readme"),
        help="Main README to write (default: the source file).",
    )
    p.add_argument(
        "--no-split",
        action="store_true",
        default=config.get("no-split", False),
        help="Append all translations to the main README instead of README.<lang>.md files.",
    )
    p.add_argument(
        "--base-url",
        default=config.get("base-url", ""),
        help="Prefix for language-bar links (e.g. for PyPI). Default: relative links.",
    )
    p.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=config.get("jobs", 1),
        help="Languages to translate in parallel (default: 1).",
    )
    p.add_argument(
        "--cache",
        default=config.get("cache", ".rosetta/cache.json"),
        help="Segment cache file; commit it to keep CI runs incremental.",
    )
    p.add_argument(
        "--no-cache", action="store_true", help="Ignore and don't write the cache."
    )
    p.add_argument(
        "--allow-fallback",
        action="store_true",
        default=config.get("allow-fallback", False),
        help="Exit 0 even if some segments could not be translated.",
    )
    p.add_argument(
        "--sphinx",
        action="store_true",
        default=config.get("sphinx", False),
        help="Set up Sphinx i18n and translate the .po catalogs.",
    )
    p.add_argument(
        "--sphinx-source",
        default=config.get("sphinx-source"),
        help="Package directory for sphinx-apidoc (default: src/ if present, else .).",
    )
    p.add_argument(
        "--gitbook",
        action="store_true",
        default=config.get("gitbook", False),
        help="Write a GitBook SUMMARY.md linking every translation.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be translated without calling the model or writing files.",
    )
    p.add_argument(
        "--list-languages", action="store_true", help="Print supported language codes."
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def resolve_langs(raw: List[str], src: str) -> List[str]:
    codes, bad = [], []
    for item in raw:
        code = normalize_code(item)
        if code is None:
            bad.append(item)
        elif code != src and code not in codes:
            codes.append(code)
    if bad:
        raise SystemExit(
            f"Unknown language code(s): {', '.join(bad)}. Run --list-languages to see the options."
        )
    return codes


def write_if_changed(path: str, content: str) -> bool:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            if f.read() == content:
                return False
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return True


def first_heading(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("#"):
            return heading_text(line)
    return ""


def report_failures(translator: Translator) -> None:
    table = Table(title="Segments left untranslated", show_lines=False)
    table.add_column("Lang")
    table.add_column("Reason")
    table.add_column("Source")
    for f in translator.failures[:50]:
        snippet = f.source if len(f.source) <= 60 else f.source[:57] + "..."
        table.add_row(f.target, f.reason, snippet)
    console.print(table)
    if len(translator.failures) > 50:
        console.print(f"... and {len(translator.failures) - 50} more")


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging(False)
    config = load_config()
    args = build_parser(config).parse_args(argv)
    setup_logging(args.verbose)

    if args.list_languages:
        for code, lang in sorted(LANGUAGES.items()):
            print(f"{code:6} {lang.name:24} {lang.native}")
        return 0

    src_lang = normalize_code(args.src_lang) or args.src_lang
    langs = resolve_langs(args.langs, src_lang)
    if not langs:
        console.print(
            "[red]No target languages. Pass --langs (e.g. --langs es fr ja) "
            "or set langs in [tool.readme-rosetta].[/red]"
        )
        return 2

    src_file = (
        os.path.join(args.path, "README.md") if os.path.isdir(args.path) else args.path
    )
    readme = args.readme or src_file
    model = args.model or DEFAULT_MODELS[args.backend]

    if args.dry_run:
        backend = Backend(model)
        backend.name = args.backend
    else:
        try:
            backend = create_backend(args.backend, model, ollama_host=args.ollama_host)
            with console.status(f"Preparing {backend.id}..."):
                backend.prepare()
        except BackendError as e:
            console.print(f"[red]{e}[/red]")
            return 1

    cache = Cache(None if args.no_cache else args.cache)
    translator = Translator(
        backend,
        source_lang=src_lang,
        cache=cache,
        glossary=config.get("glossary", {}),
        keep_terms=config.get("do-not-translate", []),
    )

    try:
        if os.path.exists(src_file):
            status = translate_readme(
                args, translator, src_file, readme, src_lang, langs
            )
            if status:
                return status
        elif not args.sphinx:
            console.print(f"[red]{src_file} not found.[/red]")
            return 1

        if args.sphinx:
            from .sphinx_handler import SphinxHandler

            if args.dry_run:
                console.print(
                    f"DRY RUN: would set up Sphinx i18n for {', '.join(langs)}"
                )
            else:
                source_dir = args.sphinx_source or (
                    "src" if os.path.isdir("src") else "."
                )
                SphinxHandler(translator).setup_sphinx(source_dir, langs, src_lang)

        if args.gitbook and not args.dry_run:
            write_summary(readme, src_lang, args.no_split)
        if not args.dry_run:
            cache.prune(langs)
    except BackendError as e:
        console.print(f"[red]{e}[/red]")
        return 1
    except KeyboardInterrupt:
        console.print(
            "[yellow]Interrupted; progress so far is saved in the cache.[/yellow]"
        )
        return 130
    finally:
        if not args.dry_run:
            cache.save()

    if translator.failures:
        report_failures(translator)
        if not args.allow_fallback:
            console.print(
                "[red]Some segments kept their source text (see above). Re-run to retry them, "
                "try a stronger --model, or pass --allow-fallback to accept.[/red]"
            )
            return 1
    return 0


def translate_readme(
    args: argparse.Namespace,
    translator: Translator,
    src_file: str,
    readme: str,
    src_lang: str,
    langs: List[str],
) -> int:
    with open(src_file, "r", encoding="utf-8") as f:
        body = strip_rosetta(f.read())

    title = first_heading(body)
    if title:
        translator.context = f'This text is from the README of the project "{title}".'
    handler = MarkdownHandler(translator)
    segments = handler.segments(body)

    if args.dry_run:
        table = Table(title=f"Dry run: {src_file}")
        table.add_column("Lang")
        table.add_column("Segments", justify="right")
        table.add_column("To translate", justify="right")
        target = "README (unified)" if args.no_split else None
        table.add_column("Output")
        for code in langs:
            table.add_row(
                code,
                str(len(segments)),
                str(translator.pending(segments, code)),
                target or translation_path(readme, code),
            )
        console.print(table)
        return 0

    console.print(
        f"Translating [bold]{src_file}[/bold] into {len(langs)} language(s) "
        f"with {translator.backend.id}"
    )
    results: Dict[str, str] = {}
    with Progress(
        TextColumn("{task.description:>8}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=False,
    ) as progress:
        tasks = {code: progress.add_task(code, total=len(segments)) for code in langs}

        def run(code: str) -> None:
            def advance(n: int) -> None:
                progress.advance(tasks[code], n)

            results[code] = handler.translate(body, code, progress=advance)
            progress.update(tasks[code], completed=len(segments))

        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            for future in [pool.submit(run, code) for code in langs]:
                future.result()

    changed = []
    if args.no_split:
        content = build_unified(body, results, readme, src_lang)
        if write_if_changed(readme, content):
            changed.append(readme)
    else:
        codes = sorted(set(discover_translations(readme)) | set(langs))
        note = GENERATED_NOTE.format(source=os.path.basename(src_file))
        for code in langs:
            path = translation_path(readme, code)
            nav = build_nav(code, codes, readme, src_lang, args.base_url)
            if write_if_changed(path, with_nav(results[code], nav, note)):
                changed.append(path)
        nav = build_nav(src_lang, codes, readme, src_lang, args.base_url)
        if write_if_changed(readme, with_nav(body, nav)):
            changed.append(readme)

    calls = translator.calls
    console.print(
        f"[green]Done.[/green] {len(changed)} file(s) updated, {calls} model request(s)."
        + ("" if changed else " Everything was already up to date.")
    )
    for path in changed:
        console.print(f"  • {path}")
    return 0


def write_summary(readme: str, src_lang: str, unified: bool) -> None:
    codes = discover_translations(readme)
    lines = [
        "# Summary",
        "",
        f"* [{get_language(src_lang).native}]({os.path.basename(readme)})",
    ]
    if not unified:
        for code in codes:
            lines.append(
                f"* [{get_language(code).native}]({os.path.basename(translation_path(readme, code))})"
            )
    write_if_changed(
        os.path.join(os.path.dirname(os.path.abspath(readme)), "SUMMARY.md"),
        "\n".join(lines) + "\n",
    )
    console.print("[green]SUMMARY.md written for GitBook.[/green]")


if __name__ == "__main__":
    sys.exit(main())
