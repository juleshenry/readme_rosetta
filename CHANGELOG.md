# Changelog

## Unreleased

Translation quality and reliability overhaul.

* Parse Markdown with markdown-it-py and translate only prose (headings, paragraphs, table cells, HTML text). Code, front matter, URLs, inline HTML, emoji and link targets are never sent to the model.
* Validate every segment (placeholders, line structure, length, output language/script) and retry rejected segments with the reason fed back to the model.
* Reject markup the model invents (HTML tags, code spans, links, headings); unwrap harmless wrappers such as a `<span>` or code fence around the whole reply.
* Report segments that could not be translated and exit non-zero (`--allow-fallback` to accept).
* Segment-level cache (`.rosetta/cache.json`) keyed on text, languages, backend, model and glossary: re-runs only translate changed paragraphs.
* New `anthropic` backend (`pip install "readme-rosetta[anthropic]"`); default Ollama model is now `qwen2.5:7b`.
* Language bar with native language names that links only to files that exist; replaces the old "Documentation Support" table, which is removed automatically.
* In-page anchor links are rewritten to point at translated headings; GitHub-compatible heading slugs.
* `do-not-translate` terms and per-language `glossary` in `[tool.readme-rosetta]`.
* `--jobs` for parallel languages, `--list-languages`, regional codes such as `pt-BR`, dry-run shows segments to translate.
* Reusable GitHub Action (`uses: juleshenry/readme_rosetta@main`) that opens a pull request.
* Fix: CLI crashed on Python < 3.11 (`tomllib`); now supports 3.9+.
* Fix: `publish.py` also rewrote ruff's `target-version` when bumping the version.
* Removed `--raw` and the implicit Spanish-only mode when no languages are given.
* Repo: stop tracking build artifacts, compiled catalogs and coverage data.

## v0.1.7 (2026-02-16)

* Automated release update

## v0.1.6 (2026-02-16)

* Automated release update

## v0.1.5 (2026-02-16)

* Automated release update

## v0.1.4 (2026-02-15)

* Automated release update

## v0.1.3 (2026-02-15)

* Automated release update

## v0.1.2 (2026-02-15)

* Automated release update

## v0.1.1 (2026-02-15)

* Automated release update

## v0.1.0 (2026-02-14)

* Initial release of Readme Rosetta.
