# 🗿 README Rosetta 

**README Rosetta** は、ローカル LLM を使用して documentation を複数言語に翻訳する強力な automationツールです。 [Ollama](https://ollama.ai/) により、プロジェクトの Worldwide アクセスを保証し、 perect Markdown フォーマットとドキュメント構造が維持されます。
## 🌍 README Translation

README Rosetta specialize in making your GitHub project international with minimal effort.

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

###
## コマンドラインインターフェイス (CLI)
コマンドラインインターフェイスは、直感的なものの強力なものになっている。
### 設立

```bash
pip install readme-rosetta
```

*ノート：OLLAMA（https://ollama.ai/）をシステム上で実行している必要があります。
### Global Options

| オプション | 説明 | デフォルト |
| :--- | :--- | :--- |
| `path` | ソースファイルまたはプロジェクトディレクトリーのパス。 | `README.md` |
| `--langs` | Target language codes (e.g., `es fr de`) のリスト。 | `[]` |
| `--src-lang` | ソース言語コード。 | `en` |
| `--model` | Ollama model ID を使用する。 | `llama3.2` |
| `--readme` | メイン出力 READMEファイルのパス。 | `README.md` |
| `--no-split` | 翻訳をシングルファイルに追加。 | `False` |
| `--dry-run` | プロセスをシミュレートする。ファイル書きをせず。 | `False` |
| `--verbose` | デバッグ用に詳細なログを有効にする。 | `False` |
## 📚 Sphinx Integration

-scale-your-documentation-to-professional-levels-with-automated-sphinx-i18n-support-

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
## 📖 GitBook Support

Easily maintain a multi-language GitBook。


- **自動的な Linking:** Introduction を main README にリンクし、各翻訳済みバージョンのリストアイテムを作成します。
- **言語名の認識:** 言語コード（`es`）を英語名（`Spanish`）に自動で解釈します。

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---
## ⚙️ 設定


```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```
### ⚠️タrouブシューティング＆リミテーション

自動翻訳に使用されるLLMsは強力ですが、複雑なSphinx/ RST環境内でのフォーマットの誤りが生じる可能性があります。
### Common Issues
- **Mismatched Backticks**: LLMs might fail to close `` ` `` or ` `` ` ` string.
- **Header Lengths**: If an LLM adds bolding (`**`) to a title, the Sphinx underline may no longer match the text length.
- **Structural Hallucinations**: The model might try to add its own summaries or "helpful" code blocks that aren't in the source.
### cleanup script
We provide a utility script to identify and clear common translation errors in your `.po` files. If a translation is cleared, Sphinx will simply fall back to the original English text for that string.

```bash
# Run the cleanup utility
python3 scripts/cleanup_translations.py
```

*Note: Always review your documentation builds. While Rosetta aims for perfection, manual correction of localized `.po` files is sometimes necessary for high-stakes documentation.*
## 📜ライセンス

このプロジェクトは、MITライセンスでライセンスされます。詳しくは[LICENSE](LICENSE)ファイルを参照してください。
