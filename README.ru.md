# 🗿 README Rosetta

**README Rosetta** является мощным инструментом automation, предназначенным для перевода документации в कई языки с помощью местных LLMs по [Ollama](https://ollama.ai/). Он обеспечивает доступность проекта для глобального аудитории при сохранении perfect Markdown-форматирования и структуры документа.

---

## 🌍 README Перевод

README Rosetta specializes in making your GitHub project international with minimal effort.

- **Мultiязычная поддержка:** Переводите `README.md` в десятки языков одновременно.
- **Столбец навигации:** Аутоматически добавляет столбец "камень" (таблицу) в начале README, позволяя пользователям быстро переключаться между языками.
- **Flexible Modes:**
    - **Split Mode (Default):** Генерирует отдельные файлы (например, `README.es.md`, `README.fr.md`) для чистой структуры проекта.
    - **Unified Mode (`--no-split`):** Приставляет все переводы в основной `README.md` файл, разделенные на HTML комментарии.
- **Применение Markdown:** Обходится умным путем с заголовками, списками и блоками кода для обеспечения преобразовавшегося результата оставался функциональным иwell-formatted.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---

## 🛠 Интерфейс командной строки (CLI)

Клиентский интерфейс предназначен для быть интуитивно понятным, но и мощным.

### Установка

```bash
pip install readme-rosetta
```

*Примечание: Requirements [Ollama](https://ollama.ai/) должны быть installed и running на вашем системе.*

### Глобальные опции

| Оptsion | Description | Default |
| :--- | :--- | :--- |
| `path` | Путь к исходному файлу или директории проекта. | `README.md` |
| `--langs` | Список целевых языков кодов (например, `es fr de`). | `[]` |
| `--src-lang` | Код исходного языка. | `en` |
| `--model` | ID модели Ollama. | `llama3.2` |
| `--readme` | Путь к основному выводу README файла. | `README.md` |
| `--no-split` | Добавлять переводы в отдельный файл. | `False` |
| `--dry-run` |_SIMулировать процесс без написания файлов. | `False` |
| `--verbose` | Включить дETAЛЕЙНОВАНИЕ для отладки. | `False` |

---

## 📚 Интеграция с Sphinx

Увеличите уровень профессионализма документации с помощью автоматизированной поддержки i18n Sphinx.

Когда вы запускаете с флагом `--sphinx`, README Rosetta:
1.  **Инициализируйте Sphinx:** Установите `docs/` directory if it doesn't exist.
2.  **Аутоматически конфигурируйте i18n:** Updating `conf.py` with the necessary `locale_dirs` и `gettext` settings.
3.  **Extract strings:** Run `gettext` to find all translatable strings in your documentation.
4.  **Преобразовывайте PO Files:** Используйте LLM для перевода `.po` files, сохраняя Sphinx-specific syntax like `:role:` or `.. directive::`.
5.  **Собрать HTML:** Аutomатически генерирует локализированную версию HTML для каждого целевого языка.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---

## 📖 Поддержка GitBook

Легко поддерживать multi-language GitBook.

`--gitbook` flag generates a `SUMMARY.md` file that maps your translated READMEs into a structure compatible with GitBook's navigation.

- **Аtomатическая связь:** Связывает Introduction к вашему основному README и создает элементы списка для каждого переведенного варианта.
- **Именование языков:** Атоматически решает код языка (например, `es`) в его полном виде (например, `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---

## ⚙️ Конфигурация

Сохраните время, defining your project defaults in `pyproject.toml`:

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```
