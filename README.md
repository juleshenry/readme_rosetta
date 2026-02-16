<!-- <Original README.md> -->
# [Documentation Support in Multiple Languages](https://github.com/juleshenry/readme_rosetta/blob/main)
| About | |
| ------ | ---- |
| English | [Link to Head of Docs](#🗿-readme-rosetta) |
| Arabic | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.ar.md#🗿-readme-rosetta) |
| Danish | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.da.md#🗿-readme-rosetta) |
| German | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.de.md#🗿-readme-rosetta) |
| Spanish | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.es.md#🗿-readme-rosetta) |
| Finnish | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.fi.md#🗿-readme-rosetta) |
| French | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.fr.md#🗿-readme-rosetta) |
| Hindi | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.hi.md#🗿-readme-rosetta) |
| Hungarian | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.hu.md#🗿-readme-rosetta) |
| Italian | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.it.md#🗿-readme-rosetta) |
| Japanese | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.ja.md#🗿-readme-rosetta) |
| Dutch | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.nl.md#🗿-readme-rosetta) |
| Norwegian | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.no.md#🗿-readme-rosetta) |
| Portuguese | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.pt.md#🗿-readme-rosetta) |
| Romanian | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.ro.md#🗿-readme-rosetta) |
| Russian | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.ru.md#🗿-readme-rosetta) |
| Swedish | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.sv.md#🗿-readme-rosetta) |
| Turkish | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.tr.md#🗿-readme-rosetta) |
| Chinese | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.zh.md#🗿-readme-rosetta) |
| hihn | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.hihn.md#🗿-readme-rosetta) |
| Afrikaans | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.af.md#🗿-readme-rosetta) |
| Belarusian | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.be.md#🗿-readme-rosetta) |
| Czech | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.cs.md#🗿-readme-rosetta) |
| Greek | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.el.md#🗿-readme-rosetta) |
| Gujarati | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.gu.md#🗿-readme-rosetta) |
| Hebrew | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.he.md#🗿-readme-rosetta) |
| Indonesian | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.id.md#🗿-readme-rosetta) |
| Kannada | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.kn.md#🗿-readme-rosetta) |
| Korean | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.ko.md#🗿-readme-rosetta) |
| Malayalam | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.ml.md#🗿-readme-rosetta) |
| Polish | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.pl.md#🗿-readme-rosetta) |
| Tamil | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.ta.md#🗿-readme-rosetta) |
| Thai | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.th.md#🗿-readme-rosetta) |
| Vietnamese | [Link to Head of Docs](https://github.com/juleshenry/readme_rosetta/blob/main/README.vi.md#🗿-readme-rosetta) |

# 🗿 README Rosetta

**README Rosetta** is a powerful automation tool designed to translate your documentation into multiple languages using local LLMs via [Ollama](https://ollama.ai/). It ensures your project is accessible to a global audience while maintaining perfect Markdown formatting and document structure.

---

## 🌍 README Translation

README Rosetta specializes in making your GitHub project international with minimal effort.

- **Multi-language Support:** Translate your `README.md` into dozens of languages simultaneously.
- **Navigation Table:** Automatically prepends a navigation "stone" (table) at the top of your README, allowing users to quickly switch between languages.
- **Flexible Modes:**
    - **Split Mode (Default):** Generates separate files (e.g., `README.es.md`, `README.fr.md`) for a clean project structure.
    - **Unified Mode (`--no-split`):** Appends all translations to the main `README.md` file, separated by HTML comments.
- **Markdown Preservation:** Intelligently handles headers, lists, and code blocks to ensure the translated output remains functional and well-formatted.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---

## 🛠 Command Line Interface (CLI)

The CLI is designed to be intuitive yet powerful.

### Installation

```bash
pip install readme-rosetta
```

*Note: Requires [Ollama](https://ollama.ai/) to be installed and running on your system.*

### Global Options

| Option | Description | Default |
| :--- | :--- | :--- |
| `path` | Path to source file or project directory. | `README.md` |
| `--langs` | List of target language codes (e.g., `es fr de`). | `[]` |
| `--src-lang` | Source language code. | `en` |
| `--model` | Ollama model ID to use. | `llama3.2` |
| `--readme` | Path to the main output README file. | `README.md` |
| `--no-split` | Append translations to a single file. | `False` |
| `--dry-run` | Simulate the process without writing files. | `False` |
| `--verbose` | Enable detailed logging for debugging. | `False` |

---

## 📚 Sphinx Integration

Scale your documentation to professional levels with automated Sphinx i18n support.

When you run with the `--sphinx` flag, README Rosetta:
1.  **Initializes Sphinx:** Sets up a `docs/` directory if it doesn't exist.
2.  **Auto-configures i18n:** Updates `conf.py` with the necessary `locale_dirs` and `gettext` settings.
3.  **Extracts Strings:** Runs `gettext` to find all translatable strings in your documentation.
4.  **Translates PO Files:** Uses the LLM to translate `.po` files, preserving Sphinx-specific syntax like `:role:` or `.. directive::`.
5.  **Builds HTML:** Automatically generates localized HTML builds for every target language.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---

## 📖 GitBook Support

Easily maintain a multi-language GitBook.

The `--gitbook` flag generates a `SUMMARY.md` file that maps your translated READMEs into a structure compatible with GitBook's navigation.

- **Automatic Linking:** Links the Introduction to your main README and creates list items for every translated version.
- **Language Names:** Automatically resolves language codes (like `es`) into their full names (like `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---

## ⚙️ Configuration

Save time by defining your project defaults in `pyproject.toml`:

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```

---

## ⚠️ Troubleshooting & Limitations

Automated translation using LLMs is powerful but can occasionally introduce formatting artifacts, especially in complex Sphinx/RST environments.

### Common Issues
- **Mismatched Backticks:** LLMs might fail to close a `` ` `` or ` `` ` ` string.
- **Header Lengths:** If an LLM adds bolding (`**`) to a title, the Sphinx underline may no longer match the text length.
- **Structural Hallucinations:** The model might try to add its own summaries or "helpful" code blocks that aren't in the source.

### Cleanup Script
We provide a utility script to identify and clear common translation errors in your `.po` files. If a translation is cleared, Sphinx will simply fall back to the original English text for that string.

```bash
# Run the cleanup utility
python3 scripts/cleanup_translations.py
```

*Note: Always review your documentation builds. While Rosetta aims for perfection, manual correction of localized `.po` files is sometimes necessary for high-stakes documentation.*

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
