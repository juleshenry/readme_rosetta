# 🗿 README Rosetta

**README Rosetta** 是一个强大的自动化工具， designed 来将您的文档转换为多语言使用局部的LLM via [Ollama](https://ollama.ai/）。它确保了您的项目在全球范围内有可接触性的同时，保持了完美的Markdown格式和文档结构。

---
## 🌍 README_translation

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
## ⚙️ 命令行界面（CLI）

命令行界面设计以直观易用为主，同时具备强大的功能。
### 安装

```bash
pip install readme-rosetta
```

*注解：需要在系统上安装并运行[Ollama](https://ollama.ai/)。
### 全局选项

| 选项 | 描述 | 默认值 |
| :--- | :--- | :--- |
| `path` | 来源文件或项目目录的路径。 | `README.md` |
| `--langs` | 目标语言代码列表（例如，`es fr de`）。 | `[]` |
| `--src-lang` | 来源语言代码。 | `en` |
| `--model` | 使用的Ollama模型ID。 | `llama3.2` |
| `--readme` | 主输出README文件的路径。 | `README.md` |
| `--no-split` |将翻译附加到一个单个文件。 | `False` |
| `--dry-run` | simulation过程不写入文件。 | `False` |
| `--verbose` | 在调试中启用详细日志。 | `False` |
## 📚 Sphinx 分析

通过自动化的Sphinx i18n支持来提升你的文档到专业水平。

5.  **生成HTML：** 自动性生成各语言的HTML版本。

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---
## 📖 GitBook Support

 Easily maintain a multi-language GitBook。


- **自动链接**：将引导到主README，并为每个翻译版本创建列表项目。
- **语言名称**：自动解析语言代码（例如`es`）转换成其全名（例如`Spanish`）。

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---
```markdown
## ⚙️ Configuration

保存时间尽量定义你的项目默认设置在`pyproject.toml`：

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
```

Please note that I've kept the markdown formatting and characters exactly as they were in the original text. If you need any further assistance, please let me know!
### ⚠️ 问题解决与限制

自主翻译使用LLM强大，但偶尔会引入格式化 artefacts，尤其在复杂的Sphinx/RST环境中。
### 常见问题
- `` ` ``` or ` ``` ` `
- 头部长度：如果模型添加了粗体（`**`），Sphinx的下划线可能不再匹配文本长度。
- 结构性扰动：模型可能尝试添加自己的总结或“有助”代码块，而这些代码块并未出现在原始文中。
### clean_up_script

```bash
# Run the cleanup utility
python3 scripts/cleanup_translations.py
```
## 📜 项目是使用MIT许可证进行许可 - 详见[LICENSE](LICENSE)文件。
