# 🗿 README Rosetta

**README Rosetta** ist eine mächtige Automatisierungstool, das Ihre Dokumentation in mehrere Sprachen mit lokalen LLMs über [Ollama](https://ollama.ai/) übersetzt. Es stellt sicher, dass Ihr Projekt für eine globale Zielgruppe zugänglich ist und dabei die perfekte Markdown-Formatierung und Struktur beibehält.

---

## 🌍 README Übersetzung

README Rosetta spezialisiert sich darauf, Ihr GitHub-Projekt international mit minimal Aufwand zu machen.

*   **Multi-Sprachunterstützung:** Übersetzt Ihre `README.md`-Titel in Dutzende von Sprachen gleichzeitig.
*   **Navigationstabelle:** Eine automatische Navigation "Stein" (Tabelle) vornefügt an der Oberseite Ihres READMEs, ermöglicht es den Benutzern, zwischen den Sprachen schnell zu wechseln.
*   **Flexibale Modi:**
    *   **Spaltensplitter-Modus (Standard):** Generiert separate Dateien (z.B. `README.es.md`, `README.fr.md`) für eine saubere Projektstruktur.
    *   **Unified-Modus (`--no-split`):** Alle Übersetzungen in die Hauptdatei `README.md` einfügt, getrennt durch HTML-Kommentare.
*   **Markdown-Erlaubnis:** Intelligente Behandlung von Schlagworten, Listen und Code-Blöcken, um sicherzustellen, dass der Übersetzungsprozess funktioniert und gut formatiert bleibt.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---

## 🛠 Benutzerfritz (CLI)

Die CLI ist darauf ausgelegt, sowohl intuitiv als auch mächtig zu sein.

### Installation

```bash
pip install readme-rosetta
```

*   **Hinweis:** Erfordert [Ollama](https://ollama.ai/) auf Ihrem System.

### Globale Optionen

| Option | Beschreibung | Standardwert |
| :--- | :--- | :--- |
| `path` | Pfad zur Quelldatei oder Projektverzeichnis. | `README.md` |
| `--langs` | Liste der Ziel-Sprachcodes (z.B. `es fr de`). | `[]` |
| `--src-lang` | Quellsprache-Code. | `en` |
| `--model` | Ollama-Modell-ID zu verwenden. | `llama3.2` |
| `--readme` | Pfad zur Hauptausgabereadme-Datei. | `README.md` |
| `--no-split` | Übersetzungen in eine einzelne Datei hinzufügen. | `False` |
| `--dry-run` | Prozess simulieren, ohne Dateien schreiben. | `False` |
| `--verbose` | Detaillierte Log-Dateien für Debugging aktivieren. | `False` |

---

## 📚 Sphinx Integration

Erfahren Sie, wie Ihre Dokumentation professionell gestaltet werden kann mit automatisierter Sphinx-i18n-Bildung.

Beim Betreiben mit dem `--sphinx`-Flag:

1.  **Sphinx initialisieren:** Legt eine `docs/`-Verzeichnis fest, wenn es nicht bereits existiert.
2.  **I18N-Einrichten:** Aktualisiert `conf.py` mit den erforderlichen `locale_dirs` und `gettext`-Einstellungen.
3.  **Übersetzungs-Dateien extrahieren:** Führt die Übersetzung `gettext` durch, um alle im Dokument enthaltenen Übersetzungszeilen zu finden.
4.  **PO-Dateien übersetzen:** Verwendet das LLM, um die `.po`-Dateien zu übersetzen, wobei Sphinx-Spezifische Syntax wie `:role:` oder `.. directive::` berücksichtigt wird.
5.  **HTML-ausgeben:** Generiert automatisch lokale HTML-Bauweisen für jede Ziel-Sprache.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---

## 📖 GitBook Unterstützung

Verwenden Sie die `--gitbook`-Flag, um eine multilingue GitBook zu erhalten.

*   **Automatische Verlinkung:** Verbindet den Einleitungsteil mit der Hauptreadme und erstellt Listen-Elemente für jede Übersetzungsversion.
*   **Sprachnamen:** Automatisch übersetzt Sprachcodes (z.B. `es`) in ihre vollständigen Namen (z.B. `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---

## ⚙️ Einstellung

Sparen Sie Zeit, indem Sie Ihre Projekt-Einstellungen in `pyproject.toml` definieren:

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

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
