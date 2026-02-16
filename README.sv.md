# 🗿 README Rosetta

**README Rosetta** är en mächtig automatiserings verktyg som designas att översätta din dokumentation till flera språk med lokala LLMs via [Ollama](https://ollama.ai/). Det garanterar ditt projekt är tillgänglig för en global befolkning medan man upprätthåller perfekt Markdown-formatning och dokumentstruktur.

---

## 🌍 README Översättning

README Rosetta specialiserar sig på att göra ditt GitHub-projekt internationell med minimal ansträngning.

- **Multilangage-stöd:** Översätt din `README.md` till tjugoner av språk samtidigt.
- **Navigations-tabel:** Automatiskt lägger till en navigations "sten" (tabel) på toppen av ditt README, vilket gör det möjligt för användarna att snabbt byta mellan språk.
- **Anpassningsbara modi:
    - **Splittmod (Förfall:):** Genererar separat filer (t.ex. `README.es.md`, `README.fr.md`) för en ren projektstruktur.
    - **Enhetlig mod (`--no-split`):** Försätter alla översättningar i den huvudsakliga `README.md`-filen, skilt av HTML-noteringar.
- **Markdown-bevaring:** Handlägger intelligently headers, listor och kodblock för att säkerställa det översatt outputet är funktionsdugligt och well-formatterad.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---

## 🛠 Kommandofönster (CLI)

Kommandofönsteret är designat för att vara intuitivt men samtidigt mäktig.

### Installering

```bash
pip install readme-rosetta
```

*Note: Kräver [Ollama](https://ollama.ai/) att installeras och köras på ditt system.*

### Globala alternativ

| Alternativ | Beskrivning | Standardvärde |
| :--- | :--- | :--- |
| `path` | Väg till källfil eller projektindrivare. | `README.md` |
| `--langs` | Lista med mål språk- kod (t.ex. `es fr de`). | `[]` |
| `--src-lang` | Källspråk-kod. | `en` |
| `--model` | Ollama-modell ID att använda. | `llama3.2` |
| `--readme` | Väg till den huvudsakliga utgåvan README-filen. | `README.md` |
| `--no-split` | Försätter översättningar i en enda fil. | `False` |
| `--dry-run` | Simulerar processen utan att skriva filer. | `False` |
| `--verbose` | Aktiverar detaljert loggning för felhönskning. | `False` |

---

## 📚 Sphinx Integrering

Skala dina dokument till professionella nivåer med automatiskt integrerad Sphinx i18n-stöd.

När du körs med `--sphinx`-förklaringen:
1.  **Initialisera Sphinx:** Ställer in en `docs/`- mapp om den inte redan existerar.
2.  **Automatiserad konfiguration av i18n:** Updaterar `conf.py` med de nödvändiga `locale_dirs` och `gettext`-inställningarna.
3.  **Extrahera STRINGS:** Körs `gettext` för att hitta alla översättningsbara STRING-i dina dokument.
4.  **Översätt PO-filer:** Använder LLM-överföringen av `.po`-filer, med bevarande av Sphinx-specifika syntax som `:role:` eller `.. directive::`.
5.  **Bilda HTML:** Automatiskt genererar lokaliserade HTML-byggnader för varje mål språk.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---

## 📖 GitBook Stöd

Enkelhjälpsamt hålla en multi-lingv GitBook.

Den `--gitbook`-förklaringen genererar en `SUMMARY.md`-fil som kartlägger dina översatta READMEs i en struktur anpassad till GitBooks navigering.

- **Automatiskt länkande:** Länkar introduktionen till din huvudsakliga README och skapar listitem i varje översatt version.
- **Språknamn:** Automatiskt löser språk-koderna (som `es`) ut till sina fullständiga namn (som `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---

## ⚙️ Konfiguration

Rädda tid genom att definiera ditt projekt-standard i `pyproject.toml`:

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```
