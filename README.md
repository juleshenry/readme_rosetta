<!-- <Original README.md> -->
# [Documentation Support in Multiple Languages](https://www.github.com/juleshenry/readme_rosetta)
| About | |
| ------ | ---- |
| English | [Link to Head of Docs](#readme-rosetta) |
| Spanish | [Link to Head of Docs](#readme-rosetta-español) |
| French | [Link to Head of Docs](#-french) |
| German | [Link to Head of Docs](#-german) |
| Italian | [Link to Head of Docs](#-italian) |
| Portuguese | [Link to Head of Docs](#-portuguese) |


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
- **GitBook Compatible:** Generates documentation that can be easily imported and hosted on GitBook.
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

`readme-rosetta translate --input <path_to_input_file> --output <path_to_output_file> --target <target_language>`

--input: Path to the input README.md file.
--output: Path to the output translated file.
--target: Target language for translation.

## Documentation

Comprehensive documentation is available in the `docs` directory. It is built using Sphinx and supports multiple languages.

### Building Documentation

To build the HTML documentation in English:

```bash
cd docs
pip install -r requirements.txt
make html
```

### Multilingual Support (i18n)

The documentation is prepared for internationalization. To update or build translations for specific languages (e.g., Spanish):

```bash
cd docs
# Update translation catalogs
make translate-update LANGS=es

# Build translated HTML
make translate-build LANGS=es
```

You can build all supported languages at once by running `make translate-build` without the `LANGS` argument.

#### Supported Documentation Languages
The documentation setup includes optimized search support for:
*   **Built-in Support:** English, Arabic, Danish, Dutch, Finnish, French, German, Hungarian, Italian, Japanese (via janome), Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Turkish.
*   **Additional Support:** Chinese (via jieba), Hindi.

The translated documentation will be available at `docs/build/html/<language>/`.

### GitBook Integration

Readme Rosetta is designed to work seamlessly with GitBook. Since it generates standard Markdown, you can point GitBook to your repository or the generated `README.md` to have a multi-language documentation site in minutes.

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


<!-- toc -->


# Readme Rosetta Spanish

Readme Rosetta is a tool designed to facilitate the translation of documentation, with a current focus on GitHub README.md files and Python libraries. It is fully compatible with GitBook and utilizes state-of-the-art Transformers models for high-quality translations.

## Features

- **Transformers Powered:** Uses NLLB-200 via the `transformers` library for accurate multilingual translations.
- **GitBook Compatible:** Generates documentation that can be easily imported and hosted on GitBook.
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

`readme-rosetta translate --input <path_to_input_file> --output <path_to_output_file> --target <target_language>`

--input: Path to the input README.md file.
--output: Path to the output translated file.
--target: Target language for translation.

## Documentation

Comprehensive documentation is available in the `docs` directory. It is built using Sphinx and supports multiple languages.

### Building Documentation

To build the HTML documentation in English:

```bash
cd docs
pip install -r requirements.txt
make html
```

### Multilingual Support (i18n)

The documentation is prepared for internationalization. To update or build translations for specific languages (e.g., Spanish):

```bash
cd docs
# Update translation catalogs
make translate-update LANGS=es

# Build translated HTML
make translate-build LANGS=es
```

You can build all supported languages at once by running `make translate-build` without the `LANGS` argument.

#### Supported Documentation Languages
The documentation setup includes optimized search support for:
*   **Built-in Support:** English, Arabic, Danish, Dutch, Finnish, French, German, Hungarian, Italian, Japanese (via janome), Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Turkish.
*   **Additional Support:** Chinese (via jieba), Hindi.

The translated documentation will be available at `docs/build/html/<language>/`.

### GitBook Integration

Readme Rosetta is designed to work seamlessly with GitBook. Since it generates standard Markdown, you can point GitBook to your repository or the generated `README.md` to have a multi-language documentation site in minutes.

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


<!-- toc -->


# Readme Rosetta French

Readme Rosetta is a tool designed to facilitate the translation of documentation, with a current focus on GitHub README.md files and Python libraries. It is fully compatible with GitBook and utilizes state-of-the-art Transformers models for high-quality translations.

## Features

- **Transformers Powered:** Uses NLLB-200 via the `transformers` library for accurate multilingual translations.
- **GitBook Compatible:** Generates documentation that can be easily imported and hosted on GitBook.
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

### Legacy / Single Translation
```bash
readme-rosetta en es README_SOURCE.md
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

### Multilingual Support (i18n)

The documentation is prepared for internationalization. To update or build translations for specific languages (e.g., Spanish):

```bash
cd docs
# Update translation catalogs
make translate-update LANGS=es

# Build translated HTML
make translate-build LANGS=es
```

You can build all supported languages at once by running `make translate-build` without the `LANGS` argument.

#### Supported Documentation Languages
The documentation setup includes optimized search support for:
*   **Built-in Support:** English, Arabic, Danish, Dutch, Finnish, French, German, Hungarian, Italian, Japanese (via janome), Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Turkish.
*   **Additional Support:** Chinese (via jieba), Hindi.

The translated documentation will be available at `docs/build/html/<language>/`.

### GitBook Integration

Readme Rosetta is designed to work seamlessly with GitBook. Since it generates standard Markdown, you can point GitBook to your repository or the generated `README.md` to have a multi-language documentation site in minutes.

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


<!-- toc -->


# Readme Rosetta German

Readme Rosetta is a tool designed to facilitate the translation of documentation, with a current focus on GitHub README.md files and Python libraries. It is fully compatible with GitBook and utilizes state-of-the-art Transformers models for high-quality translations.

## Features

- **Transformers Powered:** Uses NLLB-200 via the `transformers` library for accurate multilingual translations.
- **GitBook Compatible:** Generates documentation that can be easily imported and hosted on GitBook.
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

### Legacy / Single Translation
```bash
readme-rosetta en es README_SOURCE.md
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

### Multilingual Support (i18n)

The documentation is prepared for internationalization. To update or build translations for specific languages (e.g., Spanish):

```bash
cd docs
# Update translation catalogs
make translate-update LANGS=es

# Build translated HTML
make translate-build LANGS=es
```

You can build all supported languages at once by running `make translate-build` without the `LANGS` argument.

#### Supported Documentation Languages
The documentation setup includes optimized search support for:
*   **Built-in Support:** English, Arabic, Danish, Dutch, Finnish, French, German, Hungarian, Italian, Japanese (via janome), Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Turkish.
*   **Additional Support:** Chinese (via jieba), Hindi.

The translated documentation will be available at `docs/build/html/<language>/`.

### GitBook Integration

Readme Rosetta is designed to work seamlessly with GitBook. Since it generates standard Markdown, you can point GitBook to your repository or the generated `README.md` to have a multi-language documentation site in minutes.

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


<!-- toc -->


# Readme Rosetta Italian

Readme Rosetta is a tool designed to facilitate the translation of documentation, with a current focus on GitHub README.md files and Python libraries. It is fully compatible with GitBook and utilizes state-of-the-art Transformers models for high-quality translations.

## Features

- **Transformers Powered:** Uses NLLB-200 via the `transformers` library for accurate multilingual translations.
- **GitBook Compatible:** Generates documentation that can be easily imported and hosted on GitBook.
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

### Legacy / Single Translation
```bash
readme-rosetta en es README_SOURCE.md
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

### Multilingual Support (i18n)

The documentation is prepared for internationalization. To update or build translations for specific languages (e.g., Spanish):

```bash
cd docs
# Update translation catalogs
make translate-update LANGS=es

# Build translated HTML
make translate-build LANGS=es
```

You can build all supported languages at once by running `make translate-build` without the `LANGS` argument.

#### Supported Documentation Languages
The documentation setup includes optimized search support for:
*   **Built-in Support:** English, Arabic, Danish, Dutch, Finnish, French, German, Hungarian, Italian, Japanese (via janome), Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Turkish.
*   **Additional Support:** Chinese (via jieba), Hindi.

The translated documentation will be available at `docs/build/html/<language>/`.

### GitBook Integration

Readme Rosetta is designed to work seamlessly with GitBook. Since it generates standard Markdown, you can point GitBook to your repository or the generated `README.md` to have a multi-language documentation site in minutes.

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


<!-- toc -->


# Readme Rosetta Portuguese

Readme Rosetta is a tool designed to facilitate the translation of documentation, with a current focus on GitHub README.md files and Python libraries. It is fully compatible with GitBook and utilizes state-of-the-art Transformers models for high-quality translations.

## Features

- **Transformers Powered:** Uses NLLB-200 via the `transformers` library for accurate multilingual translations.
- **GitBook Compatible:** Generates documentation that can be easily imported and hosted on GitBook.
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

### Legacy / Single Translation
```bash
readme-rosetta en es README_SOURCE.md
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

### Multilingual Support (i18n)

The documentation is prepared for internationalization. To update or build translations for specific languages (e.g., Spanish):

```bash
cd docs
# Update translation catalogs
make translate-update LANGS=es

# Build translated HTML
make translate-build LANGS=es
```

You can build all supported languages at once by running `make translate-build` without the `LANGS` argument.

#### Supported Documentation Languages
The documentation setup includes optimized search support for:
*   **Built-in Support:** English, Arabic, Danish, Dutch, Finnish, French, German, Hungarian, Italian, Japanese (via janome), Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Turkish.
*   **Additional Support:** Chinese (via jieba), Hindi.

The translated documentation will be available at `docs/build/html/<language>/`.

### GitBook Integration

Readme Rosetta is designed to work seamlessly with GitBook. Since it generates standard Markdown, you can point GitBook to your repository or the generated `README.md` to have a multi-language documentation site in minutes.

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
