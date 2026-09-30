# 🗿 README Rosetta

**README Rosetta** translates your README (and your Sphinx docs) into dozens of languages with a local model via [Ollama](https://ollama.com/) or with Claude. It parses your Markdown instead of trusting the model with it, so code blocks, tables, links and badges come through untouched, and it checks every translation before writing it.

```bash
pip install readme-rosetta
readme-rosetta --langs es fr ja
```

That writes `README.es.md`, `README.fr.md` and `README.ja.md`, and adds a language bar to the top of every file:

🌐 **English** · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md)

---

## ✨ Why it works

- **Only prose is translated.** The document is parsed with a real Markdown parser. Headings, paragraphs, table cells and the text inside HTML blocks are translated; code, front matter, URLs, inline HTML and link targets never reach the model.
- **Every translation is checked.** A segment is rejected if a placeholder goes missing, if a single line turns into several, if the length is implausible, or if the text is in the wrong language or script (for example Thai showing up in an Arabic README). Rejected segments are retried with the reason fed back to the model.
- **Failures are loud.** If a segment still fails, it keeps its source text, the run lists it, and the command exits with a non-zero status so CI notices.
- **Incremental by default.** Translations are cached per segment in `.rosetta/cache.json`. When you edit one paragraph, only that paragraph is sent to the model again, and unchanged text stays byte-for-byte identical, so diffs stay small. Commit the cache file.
- **Links keep working.** In-page links such as `[Install](#install)` are pointed at the translated headings, and the language bar only lists files that exist.

---

## 🚀 Usage

### Backends

| Backend | Setup | Default model |
| :--- | :--- | :--- |
| `ollama` (default) | Install [Ollama](https://ollama.com/download) and keep it running. The model is pulled on first use. | `qwen2.5:7b` |
| `anthropic` | `pip install "readme-rosetta[anthropic]"` and set `ANTHROPIC_API_KEY`. | `claude-opus-5-5` |

Small models are fast but make more mistakes, especially in languages that do not use the Latin alphabet. If the run reports failures, use a larger model or the `anthropic` backend.

```bash
# Local, with a bigger model
readme-rosetta --langs ar hi ja --model qwen2.5:14b

# Claude, four languages at a time
readme-rosetta --backend anthropic --langs de es fr it ja ko pt-BR zh --jobs 4
```

### Options

| Option | Description | Default |
| :--- | :--- | :--- |
| `path` | Source Markdown file, or a directory that contains `README.md`. | `README.md` |
| `--langs` | Target language codes, such as `es fr pt-BR`. Run `--list-languages` to see all of them. | from config |
| `--src-lang` | Source language code. | `en` |
| `--backend` | `ollama` or `anthropic`. | `ollama` |
| `--model` | Model name for the backend. | see above |
| `--jobs`, `-j` | Number of languages to translate in parallel. | `1` |
| `--readme` | Main README to write, if it is not the source file. | source file |
| `--no-split` | Append every translation to the main README instead of writing `README.<lang>.md` files. | off |
| `--base-url` | Prefix for language bar links, for places where relative links break (such as PyPI). | relative links |
| `--cache` | Cache file. | `.rosetta/cache.json` |
| `--no-cache` | Ignore the cache and don't write it. | off |
| `--allow-fallback` | Exit with status 0 even if some segments could not be translated. | off |
| `--dry-run` | Show how many segments each language needs, without calling the model or writing files. | off |
| `--verbose`, `-v` | Detailed logging. | off |

### Configuration

Put your defaults in `pyproject.toml`, then run `readme-rosetta` with no arguments:

```toml
[tool.readme-rosetta]
langs = ["es", "fr", "ja", "pt-BR"]
backend = "ollama"
model = "qwen2.5:7b"
jobs = 2

# Never translated: product names, commands, brand terms
do-not-translate = ["README Rosetta", "Ollama"]

# Preferred translations for recurring terms, per language
[tool.readme-rosetta.glossary.es]
"pull request" = "solicitud de cambios"
"release" = "versión"
```

---

## 🤖 GitHub Action

Keep translations up to date automatically. On every push that changes the README, this workflow re-translates the changed paragraphs and opens a pull request:

```yaml
name: Translate README
on:
  push:
    branches: [main]
    paths: [README.md]
  workflow_dispatch:

permissions:
  contents: write
  pull-requests: write

jobs:
  translate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: juleshenry/readme_rosetta@main
        with:
          langs: es fr ja
          backend: anthropic
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

Leave out `backend` to run Ollama on the runner instead. That needs no API key, but it is slow on a CPU runner.

---

## 📚 Sphinx Integration

With `--sphinx`, README Rosetta sets up Sphinx internationalization and translates the documentation:

1. **Sets up Sphinx:** creates `docs/` if it doesn't exist.
2. **Configures i18n:** adds `locale_dirs` and `gettext_compact` to `conf.py`.
3. **Extracts strings:** runs `gettext` and `sphinx-intl` to create `.po` catalogs.
4. **Translates catalogs:** translates new and fuzzy entries, keeping reStructuredText roles, literals, links and substitutions intact.
5. **Builds HTML:** builds the site once per language.

```bash
readme-rosetta --sphinx --langs es ja
```

Entries that fail validation are left empty, so Sphinx falls back to the source text for them. To clear bad entries from older catalogs, run `python3 scripts/cleanup_translations.py`.

---

## 📖 GitBook

`--gitbook` writes a `SUMMARY.md` that links your README and every translation, for GitBook navigation.

```bash
readme-rosetta --gitbook --langs hi zh pt
```

---

## 🔄 Upgrading from 0.1.x

- The old "Documentation Support in Multiple Languages" table is removed automatically on the first run and replaced by the language bar.
- Old cache files are ignored. The first run translates everything again.
- The default Ollama model is now `qwen2.5:7b`. To keep the old model, set `model = "llama3.2"`.
- `--raw` has been removed. Language bar links are relative unless you pass `--base-url`.

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
