# 🗿 README Rosetta

**README Rosetta** is een krachtige automatiserings-tool om uw documentatie te vertalen naar meerdere talen met lokale LLMs via [Ollama](https://ollama.ai/). Het zorgt ervoor dat uw project toegankelijk is voor een wereldwijde publieke terwijl het de perfecte Markdown-vormaatting en documentstructuur behoudt.

---

## 🌍 README Vertaling

README Rosetta specialiseert zich in het maken van uw GitHub-projekt internationaal met minimale inspanning.

- **Meer dan een taal:** Vertaal uw `README.md` naar tientallen talen tegelijk.
- **Navigatietabel:** Automatisch voegt een navigatie "steen" (tabel) toe aan het begin van uw README, waardoor gebruikers sneller tussen de talen kunnen springen.
- **Flexibele modi:**
    - **Splitsch Modus (Standards):** Genereert aparte bestanden (bijv. `README.es.md`, `README.fr.md`) voor een frisse projectstructuur.
    - **Geïntegreerde Modus (`--no-split`):** Voegt alle vertalingen toe aan het hoofdbestand `README.md`, gescheiden door HTML-opmerkingen.
- **Markdown-beheer:** Intelligent behandelt headers, lijsten en codeblokken om zeker te stellen dat de vertaalde uitvoering functioneel en goed vormaat blijft.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---

## 🛠 Bevelsysteem (CLI)

Het bevelsysteem is ontworpen om intuïtief en krachtig te zijn.

### Installatie

```bash
pip install readme-rosetta
```

*Nota: Voordat u begint, moet [Ollama](https://ollama.ai/) op uw systeem worden geïnstalleerd.*

### Globale opties

| Optie | Beschrijving | Standaard |
| :--- | :--- | :--- |
| `path` | Pad van bronbestand of projectverbinding. | `README.md` |
| `--langs` | Lijst van doeltaalcodes (bijv. `es fr de`). | `[]` |
| `--src-lang` | Bron-taalcode. | `en` |
| `--model` | ID Ollama-modelfout voor gebruik. | `llama3.2` |
| `--readme` | Pad van de hoofdoutput-README-bestand. | `README.md` |
| `--no-split` | Vertaal bestanden naar één bestand. | `False` |
| `--dry-run` | Simuleer het proces zonder bestanden te schrijven. | `False` |
| `--verbose` | Activeren van gedetailleerde loggering voor debuggen. | `False` |

---

## 📚 Sphinx-integratie

Schaal uw documentatie uit tot professionele niveaus met automatische Sphinx i18n-steun.

Wanneer u met de `--sphinx`-vlag uitvoert, README Rosetta:
1.  **Initialiseert Sphinx:** Maakt een `docs/`-verbinding aan als het bestaat niet.
2.  **Auto-configureert i18n:** Updatet `conf.py` met de benodigde `locale_dirs` en `gettext`-instellingen.
3.  **Extracteert strings:** Voert `gettext` uit om alle vertaalbare strings in uw documentatie te vinden.
4.  **Vertaalt PO-bestanden:** Gebruikt het LLM om `.po`-bestanden te vertalen, de Sphinx-specifieke syntax zoals `:role:` of `.. directive::` te behouden.
5.  **Genereert HTML:** Automatisch genereren van gelocaliseerde HTML-pagina's voor elke doeltaal.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---

## 📖 GitBook-steun

Eenvoudig onderhoud uw multi-taal GitBook.

De `--gitbook`-vlag generëert een `SUMMARY.md`-bestand dat de vertaalde README's in een structuur compatibel met GitBooks navigatie geeft.

- **Automatische linking:** Koppelt het Inleiding naar uw hoofdREADME en creëert list-items voor elke vertaalde versie.
- **Taalnamen:** Automatisch resolvereer taalcodes (bijv. `es`) in hun volle namen (bijv. `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---

## ⚙️ Configuratie

Spare tijd door uw project-instellingen voor te definiëren in `pyproject.toml`:

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```
