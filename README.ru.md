# 🗿 ПРЕЗАТЧИК Rosetta

**ПРЕЗАТЧИК Rosetta** — мощный инструмент автоматизации, который обеспечивает перевод документации для различных языков utilizing local LLMs via [Ollama](https://ollama.ai/). Это ensures your project is accessible to a global audience while maintaining perfect Markdown formatting and document structure.
## 🌍 README Перевод

README Rosetta specializirovalosia v prodvizhenii vseh vashej GitHub projekts mezhdushnie s minimumi effortsom.

- **Mnogovzyrnye otchestva:** Perevedeny vos' `README.md` na desyatki yezov. 
- **Nabroskaya tablitsa:** Avtomatichno prepredislyaet kapushku "stony" (tablitsu) v gorney rosti vashego README, allowing users to quickly switch between languages.
- **Flexibil'nye moodi:**
    - **Razdelenny mode (Poobshenniy):** Vyzhdut otrazhenie separate filey (etot `README.es.md`, `README.fr.md`) dlya chistogo struktury projekta.
    - **Svoinzirannyi mode (`--no-split`):** Perevedeny vos' vse all translations to the main `README.md` file, separated by HTML comments.
- **Mnogotipnye zapisy:** Intelligently handles headers, lists, and code blocks to ensure the translated output remains functional and well-formatted.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```
## 🔧 Клиентский интерфейс командной строки (КLI)

КЛИ проектируется с учетом интуитивного и мощного дизайна.
### Installация

```bash
pip install readme-rosetta
```

*Примечание.Необходим[Оллама](https://ollama.ai/).Чтобы запустить,需 установить и запустить Олламу на своем устройстве.*
### Глобальные настройки

| Опция | Beschreibung | Предварительный значения |
| :--- | :--- | :--- |
| `path` | Путь к файлу или директории источного проекта. | `README.md` |
| `--langs` | Список кодов цели (например, `es fr de`). | `[]` |
| `--src-lang` | Код языка источника. | `en` |
| `--model` | ID модели Ollama, используемого. | `llama3.2` |
| `--readme` | Путь к основному README с переводами. | `README.md` |
| `--no-split` | Добавлять переводы вsingle файл. | `False` |
| `--dry-run` | Симулировать процесс без записи файлов. | `False` |
| `--verbose` | Включить详ный логи для отладки. | `False` |
## 📚 Integratsiya Sphinx

Повышайте качество документации до профессиональных уровней с использованием автоматизированной поддержки Sphinx i18n.

Когда вы запускаете с флагом `--sphinx`, README Rosetta:
1.  **Инициализируйте Sphinx:** Установите директорию `docs/`, если она नह существует.
2.  **Автоматически конфигурируйте i18n:** Обновляйте `conf.py` с necessary `locale_dirs` и `gettext` настройками.
3.  **Вытащите изделия:** Запустите `gettext`, чтобы найти все переводимые строки в вашей документации.
4.  **Переведите файлы PO:** Используйте LLM для перевода `.po` файлов, сохраняя Sphinx-специфические синтаксис, как `:role:` или `.. directive::`.
5.  **Строите HTML:** Аutomатически генерируйте localized HTML-построения для каждой цели языка.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---
## 📖 Поддержка GitBook

Легко поддерживать мультиязычный GitBook.

Флаг `--gitbook` генерирует файл `SUMMARY.md`, которыйитtranslated READMEs в структуру совместимую с навигацией в GitBook's navigation.

- **Автоматическое_linkirovaniye:** Связывает Introduction со своей основной README и создает элементы списка для каждого переведенного варианта.
- **Имя языка:** Авtomатически resolveет коды языков (как `es`) в их полные названия (как `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---
## ⚙️ Configурация

Сохраните время, определяя defaults для своего проекта в `pyproject.toml`:

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
⚠️ ТROБОЙСКАЯ ДЕНЕГИРУЮЩАЯ РАБОТА с ГРУППОВОК АННОТАЦИЙ может быть мощным, но в некоторых случаях она может вносить форматные artefакты, особенно в сложных средах Sphinx/RST.
### Общие Probleмы
- **Случай не闭ения тегов**: модели LLM могут не закрывать `` `` `` or `` `` `` строку.
- **Длина заголовков**: если модель добавляетbolding (`**`) к заголовку, подчеркивание Sphinx может больше не соответствовать длине текста.
- **СтрUCTУРНЫЕ Г haluccinations**:(model может попробовать добавить свои собственные обзоры или "помощительные" блоки кода, которые нет в исходном тексте).
### Скрипт очистки ошибок перевода
Мы предлагаем utility-сクリпт, который выявляет и оценивает общие ошибки перевода в ваших `.po` файлах. Если перевод cleared, Sphinx просто будет fallback'ировать на исходный английский текст для этой строки.

```bash
# Run the cleanup utility
python3 scripts/cleanup_translations.py
```

*Примечание: Always review your documentation builds. While Rosetta aims for perfection, manual correction of localized `.po` files is sometimes necessary for high-stakes documentation.*
## 📜 Лाइसенция

Этот проект лицензируется под сендером MIT - см. soubor [LICENSE](LICENSE).
