# 

**README Rosetta** は、ドキュメントをさまざまな言語に翻訳するための強力なオートメーションツールです。これは、ローカルLLMsの[Ollama](https://ollama.ai/)を使用して、そのプロジェクトが世界的的なユーザーにアクセスできるようにします。また、マークダウン形式のドキュメントの構造とフォーマットを完全に保持することを保証します。

---

## 

README Rosettaは、GitHubプロジェクトを国際化するための最適なツールです。

- **モードの flexibility:**
    - **split mode (デフォルト):** separate files (例えば`README.es.md`, `README.fr.md`) を生成してプロジェクト構造をきれいに維持します。
- **Markdownの保守:** ヘッダー、リスト、コードブロックなどのマークダウン形式を含むドキュメントの機能を完全に維持します。

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---

## 

CLIは、ユーザーriendlyかつ強力なです。

### インストール

*注目: [Ollama](https://ollama.ai/)がシステム上で実行されなければならないことに注意してください.*

### 全体的なオプション

| オプション | 説明 | デフォルト |
| :--- | :--- | :--- |
| `path` | ソースファイルまたはプロジェクトディレクティリーやローカルパス | `README.md` |
| `--langs` | target language codes (例:`es fr de`) | `[]` |
| `--src-lang` | ソース言語コード | `en` |
| `--model` | OllamaモデルのIDを使用する | `llama3.2` |
| `--readme` | 主な出力READMEファイルのパス | `README.md` |
| `--no-split` | トランスラテーションをシングルファイルに追加 | `False` |
| `--dry-run` | プロセスをシミュレートする (出力はファイルに書き込まれない) | `False` |
| `--verbose` | ディタールなロギングを有効にする | `False` |

---

## 

Sphinxの統合により、プロフェッショナルレベルまでドキュメントをスケールすることができます。

5.  **HTMLを生成します:** オートメーションしたローカライズされたHTMLビルドを作成します。

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---

## 

GitBookのサポートにより、多言語GitBookを簡単に維持することができます。

- **自動Linking:** Introduction から README にリンクし、各翻訳されたバージョンのリストアイテムを作成します。
- **language Names:** languageコード (例:`es`) を language_nameに解釈することで、タイトルとアイコンを変更できます。

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---

## 

構成を簡素化するために、プロジェクトのデフォルト設定を定義します。

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

## 

ライセンスはMIT Licenseです - それぞれ[LICENSE](LICENSE)ファイル参照ください。
