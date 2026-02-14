
<!-- <Original README.md> -->
# [Documentation Support in Multiple Languages](https://www.github.com/juleshenry/readme_rosetta)
| About | |
| ------ | ---- |
| English | [Link to Head of Docs](#readme-rosetta) |
| Spanish | [Link to Head of Docs](#readme-rosetta-español) |
# Readme Rosetta

Readme Rosetta is a tool designed to facilitate the translation of documentation, with a current focus on GitHub README.md files and Python libraries. It is fully compatible with GitBook and utilizes state-of-the-art Transformers models for high-quality translations.

## Features

- **Transformers Powered:** Uses NLLB-200 via the `transformers` library for accurate multilingual translations.
- **GitBook Integration:** Use the `--gitbook` flag to generate a `SUMMARY.md` file for seamless GitBook hosting.
- **Bulk Translation:** Support for translating into 30+ languages in a single pass.
- **Sphinx Integration:** Automatically handles Sphinx i18n workflows.
- **Universal Documentation:** Makes Python libraries and READMEs accessible world-wide.

## Getting Started

### Installation

To install Readme Rosetta, follow these steps:

```bash
pip install readme-rosetta
```

## Usage

### Sphinx Documentation Setup & Translation
To automatically setup Sphinx, generate API documentation, and translate all `.po` files for multiple languages:

```bash
readme-rosetta . --sphinx --langs es fr hi de it ja ko ru
```

### Bulk README Translation
To translate your `README_SOURCE.md` into multiple languages and append them to `README.md` with a language selector table:

```bash
readme-rosetta README_SOURCE.md --langs es fr hi de it ja
```

### GitBook Support
To generate a `SUMMARY.md` file along with your translated README for GitBook compatibility:

```bash
readme-rosetta README_SOURCE.md --langs es fr hi --gitbook
```

## Documentation

Comprehensive documentation is available in the `docs` directory. It is built using Sphinx and supports multiple languages.

### Building Documentation

To build the HTML documentation in English:

```bash
cd docs
pip install -r requirements.txt
make html
```

### Building Translated Documentation
After running the translation with `--sphinx`, you can build the HTML for a specific language using:

```bash
cd docs
# Build Spanish translation
make html SPHINXOPTS="-D language='es'"

# Or using sphinx-build directly
python -m sphinx.cmd.build -b html source build/html/es -D language='es'
```

#### Supported Documentation Languages
The documentation setup includes optimized search support for:
*   **Built-in Support:** English, Arabic, Danish, Dutch, Finnish, French, German, Hungarian, Italian, Japanese (via janome), Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Turkish.
*   **Additional Support:** Chinese (via jieba), Hindi.

The translated documentation will be available at `docs/build/html/<language>/`.

### GitBook Integration

Readme Rosetta is designed to work seamlessly with GitBook. By using the `--gitbook` flag, the tool generates a `SUMMARY.md` file that GitBook uses to build its navigation sidebar. This allows you to have a multi-language documentation site in minutes.

Simply point GitBook to your repository, and it will use the generated `README.md` and `SUMMARY.md`.

### Supported Languages (30+)

The following languages are supported and tested:
- en - English
- es - Spanish
- fr - French
- de - German
- it - Italian
- pt - Portuguese
- ru - Russian
- zh - Chinese
- ja - Japanese
- ar - Arabic
- hi - Hindi
- bn - Bengali
- id - Indonesian
- tr - Turkish
- vi - Vietnamese
- pl - Polish
- nl - Dutch
- sv - Swedish
- no - Norwegian
- da - Danish
- fi - Finnish
- el - Greek
- cs - Czech
- hu - Hungarian
- ro - Romanian
- uk - Ukrainian
- th - Thai
- ko - Korean
- he - Hebrew
- fa - Persian
- ms - Malay

## Translator Backend
Readme Rosetta now defaults to using the **NLLB-200** model via the `transformers` library ([facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)).

To use this:
1. Ensure the required packages are installed:
   ```bash
   pip install transformers torch sentencepiece protobuf
   ```
2. The model will be automatically downloaded from Hugging Face on first run.


# Contributing
We welcome contributions! If you'd like to contribute, please see our contributing guidelines.

# Bug Reporting and Support
If you encounter any issues or need support, please open an issue.

# License
This project is licensed under the MIT License.
