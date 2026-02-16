# 🗿 README Rosetta

**README Rosetta** ist ein mächtiges Automatisierungs-Tool, das Ihre Dokumentation in mehrere Sprachen übersetzt und local LLMs über [Ollama](https://ollama.ai/) verwendet. Es gewährleistet, dass Ihr Projekt für eine globale Zielgruppe zugänglich ist, während der Markdown-Formatierung und die Dokumentstruktur perfekt erhalten bleiben.

---

## 🌍 README Übersetzung

README Rosetta spezialisiert sich darauf, Ihre GitHub-Projekt international zu machen mit minimalem Aufwand.

- **Vielfältige Sprachunterstützung:** Übersetzen Sie Ihre `README.md` in Dutzende von Sprachen gleichzeitig.
- **Navigationstabelle:** Automatisch vorverlegt eine Navigation "Stein" (Tabelle) an der Spitze Ihres README, sodass Benutzer schnell zwischen den Sprachen wechseln können.
- **Flexiblere Modi:**
    - **Split-Modus (Standards):** Generiert separate Dateien (z. B. `README.es.md`, `README.fr.md`) für eine saubere Projektstruktur.
    - **Unified-Modus (`--no-split`):** Fügt alle Übersetzungen dem Haupt-`README.md`-Datei hinzu, getrennt durch HTML-Kommentare.
- **Markdown-Preservation:** Intelligente Verarbeitung von Überschriften, Listen und Codeblöcken, um sicherzustellen, dass der übersetzte Ausgangsbetrag funktional und gut gestaltet ist.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---

## 🛠 Benutzeroberfläche (CLI)

Die CLI ist darauf ausgelegt, intuitsiv aber leistungsfähig zu sein.

### Installation

```bash
pip install readme-rosetta
```

*Note: Erfordert [Ollama](https://ollama.ai/) auf Ihrem System auszuführen.*

### globale Optionen

| Option | Beschreibung | Standardwert |
| :--- | :--- | :--- |
| `path` | Pfad zur Quelldatei oder Projektverzeichnis. | `README.md` |
| `--langs` | Liste der Ziel-Sprachcodes (z. B. `es fr de`). | `[]` |
| `--src-lang` | Quellensprachen-Code. | `en` |
| `--model` | Ollama-Modell-ID, das verwendet werden soll. | `llama3.2` |
| `--readme` | Pfad zur Hauptausgabe-README-Datei. | `README.md` |
| `--no-split` | Übersetzungen auf eine einzelne Datei anhängen. | `False` |
| `--dry-run` | Simulation des Vorgangs ohne Dateien schreiben. | `False` |
| `--verbose` | Einstellungen für detaillierte Log-Dateien zum Debuggen aktivieren. | `False` |

---

## 📚 Sphinx Integration

Erhöhen Sie Ihre Dokumentation auf professionelle Stufe mit automatischer Sphinx i18n-Support.

Beim Ausführen mit dem `--sphinx`-Flag:
1.  **Initialisierung von Sphinx:** Erstellt eine `docs/`-Verzeichnis, wenn es nicht existiert.
2.  **Auto-Konfiguration von i18n:** Aktualisiert die `conf.py`-Einstellungen mit den notwendigen Einstellungen für `locale_dirs` und `gettext`.
3.  **Ausweisung von Übersetzungszeichen:** Laufen Sie `gettext`, um alle übersetzbarischen Zeichen in Ihrer Dokumentation zu finden.
4.  **Übersetzung PO-Dateien:** Verwendet die LLM, um `.po`-Dateien zu übersetzen, die Sphinx-Spezifische Syntax wie `:role:` oder `.. directive::` verwenden.
5.  **Erstellung von HTML:** Generiert automatisch lokalisierte HTML-Bausteine für jede Ziel-Sprache.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---

## 📖 GitBook Support

Einfach Ihre multilinguale GitBook halten können.

Der `--gitbook`-Flag generiert einen `SUMMARY.md`-Datei, die eine Übersetzungsstruktur kompatibel mit GitBooks Navigation erstellt.

- **Automatische Verknüpfung:** Verknüpft die Einführung mit Ihrer Haupt-README und schafft Listenpunkte für jede übersetzte Version.
- **Sprachnamen:** Automatisch löst Sie den Sprachcoden (z. B. `es`) in seine volle Bezeichnung (z. B. `Spanish`) auf.

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---

## ⚙️ Konfiguration

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
