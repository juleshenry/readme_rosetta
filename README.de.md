# 🗿 README Rosetta

**README Rosetta** ist eine mächtige Automatisierungssoftware, die Ihre Dokumentation in mehrere Sprachen übersetzt und ihre lokale LLM über [Ollama](https://ollama.ai/) verwendet. Sie gewährleistet, dass Ihr Projekt für eine globale Zielgruppe zugänglich ist und gleichzeitig perfekte Markdown-Formatierung und -Struktur beibehält.
## 🌍 README Übersetzung

README Rosetta spezialisiert sich darauf, Ihr GitHub-Projekt international mit minimalem Aufwand zu machen.

- **Multilinguale Unterstützung:** Übersetzt `README.md` in Dutzende von Sprachen gleichzeitig.
- **Navigationstabelle:** Automatisch vornimmt eine Navigation "Stein" (Tabelle) an der Spitze Ihres README, ermöglicht es den Nutzern, schnell zwischen Sprachen zu wechseln.
- **Anpassbare Modi:
    - **Spaltmodus (Standalone):** Generiert getrennte Dateien (z.B. `README.es.md`, `README.fr.md`) für eine saubere Projektstruktur.
    - **Einheitsmodus (`--no-split`):** Fügt alle Übersetzungen dem Haupt-`README.md`-File hinzu, getrennt durch HTML-Kommentare.
- **Markdown-Erhaltung:** Versteht Intelligenter Header, Liste und Codeblock, um den übersetzten Ausgang einheitlich zu machen und gleichzeitig benutzerfreundlich.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---
## 🔧 Befehlszeile (Befehlslinie)

Die Befehlszeile wird darauf ausgelegt, intuitiv noch mächtig zu sein.
### Installation

```bash
pip install readme-rosetta
```

*Hinweis: Erfordert [Ollama](https://ollama.ai/) zu sein, das auf Ihrem System ausgeführt werden muss.*
### Globale Optionen

| Option | Beschreibung | Standardwert |
| :--- | :--- | :--- |
| `path` | Pfad zur Quelldatei oder Projektverzeichnis. | `README.md` |
| `--langs` | Liste der Zielsprachencodes (z.B. `es fr de`). | `[]` |
| `--src-lang` | Quellensprachencode. | `en` |
| `--model` | Ollama-ModellID zum verwenden. | `llama3.2` |
| `--readme` | Pfad zur Hauptoutput-README-Datei. | `README.md` |
| `--no-split` | Übersetzungen zu einem einzigen Datei hinzufügen. | `False` |
| `--dry-run` | Simuliere den Prozess ohne Dateien schreiben. | `False` |
| `--verbose` | Aktiviere detaillierte Log-Verfolgung für Debugging. | `False` |
## 📚 Sphinx Integration

Skaliere deine Dokumentation auf professionelle Ebenen mit automatisierter Sphinx-i18n-Betreuung.

Beim Ausführen mit der `--sphinx`-Flag wird README Rosetta:
1.  **Initialisiert Sphinx:** Setzt das `docs/`-Verzeichnis, wenn es nicht existiert.
2.  **Auto-Konfiguriert i18n:** Aktualisiert die `conf.py`-Einstellungen mit den notwendigen `locale_dirs` und `gettext`-Einstellungen.
3.  **Extrahiert Strings:** Läuft `gettext`, um alle übersetzbarkeitsspezifischen Strings in deiner Dokumentation zu finden.
4.  **Übersetzt PO-Dateien:** Verwendet das LLM, um `.po`-Dateien zu übersetzen, die den Sphinx-Syntax wie `:role:` oder `.. directive::` beachten.
5.  **Baut HTML-Abläufe:** Erzeugt automatisch lokalisierte HTML-Abläufe für jede Ziel Sprache.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---
## 📖 GitBook Support

Einfach eine multilinguale GitBook halten.

Die `--gitbook`-Flag generiert ein `SUMMARY.md`-File, das deine Übersetzungen in eine Struktur umwandelt, die mit der Navigation von GitBook kompatibel ist.

*   **Automatischer Linking:** Verbindet den Einführungsteil mit deiner HauptREADME und erstellt Liste-Einträge für jede Übersetzungsversion.
*   **Sprachnamen:** Löst automatisch Sprachen codes (wie `es`) in ihre vollständigen Namen (wie `Spanish`) auf.

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---
## ⚙️ Konfiguration

Speichere Zeit und definier deine Projektstandards in `pyproject.toml`:

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```
⚠️ Troubleshooting & Begrenzungen

Automisierte Übersetzung mit LLMs ist leistungsfähig aber kann gelegentlich Formattierungsfehler im komplexen Sphinx/RST-Bereich vornehmen.
### Common Issues
- **Mismatched Backticks:** LLMs könnten versagen, einen `` `` `` or `` `` ``-String nicht zu schließen.
- **Header Lengths:** Wenn ein LLM eine Titelunterbrechung (`**`) hinzufügt, kann die Sphinx-Titelhöhe nicht mehr der Textlänge entsprechen.
- **Strukturhalluzinationen:** Das Modell könnte eigene Zusammenfassungen oder "hilfsvolle" Codeblöcke hinzufügen, die im Quelltext nicht enthalten sind.
### Reinigungs-Script
Wir bieten eine Benutzer-Utility-Skript zur Erkennung und Löschung häufiger Fehler der Übersetzungen in Ihren `.po`-Dateien. Wenn eine Übersetzung gelöscht wird, verwendet Sphinx einfach die ursprüngliche englische Textdatei für diese Zeile.

```bash
# Run the cleanup utility
python3 scripts/cleanup_translations.py
```

*Hinweis: Besuchen Sie Ihre Dokumentation immer noch. Rosetta strebt nach Perfektion, aber die manuelle Korrektur der lokalisierten `.po`-Dateien ist manchmal für hochkritische Dokumentation notwendig.*
## 📜 Lizenz

Dieses Projekt ist unter der GNU General Public License - siehe die [LICENSE](LICENSE) Datei für Details.
