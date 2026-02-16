# 🗿 README Rosetta

**README Rosetta** is een krachtige automatiserings tool ontworpen om je documentatie te vertalen naar meerdere talen met plaatselijke LLM's via [Ollama](https://ollama.ai/). Het zorgt ervoor dat je project toegankelijk is voor een globaal publiek terwijl de perfecte Markdown-formaat en documentstructuur worden behouden.
## 🌍 README Oplezing

README Rosetta specializes in making your GitHub project international met minimale moeite.

- **Meer dan één taal ondersteuning:** Vertaal `README.md` naar tientallen talen tegelijk.
- **Navigatietabel:** Automatisch voegt een navigatie "steen" (tabel) toe aan de top van uw README, waardoor gebruikers snel tussen talen kunnen switchen.
- **Flexibele modi:**
    - **Verdeling mode (Standards):** Genereert afzonderlijke bestanden (bijv. `README.es.md`, `README.fr.md`) voor een schoon projectstructuur.
    - **Geïntegreerde modus (`--no-split`):** Voegt alle vertalingen toe aan de hoofdbestand `README.md`, gescheiden door HTML opmerkingen.
- **Markdownbehoud:** Intelligent hanvelt headers, lijsten en codeblokken om te zorgen dat de vertaalde uitvoering functioneel en goed gevormd blijft.
## 🛠 Beveling CLI

De CLI is ontworpen om bedoeld te zijn intuitief nog steeds krachtig.
### Installatie
```bash
pip install readme-rosetta
```

*Opmerking: Vereist [Ollama](https://ollama.ai/) te installeren en te lopen op uw systeem.*
### Globale opties

| Optie | Beschrijving | Standaardwaarde |
| :--- | :--- | :--- |
| `path` | Pad om naar bronbestand of projectdirectory te verwijzen. | `README.md` |
| `--langs` | lijst met doeltaalcodes (bijv. `es fr de`). | `[]` |
| `--src-lang` | Bronstalencode. | `en` |
| `--model` | Ollama model ID om te gebruiken. | `llama3.2` |
| `--readme` | Pad naar het hoofdoutput README bestand. | `README.md` |
| `--no-split` | Vertaal vertalingen toe aan een enkel bestand. | `False` |
| `--dry-run` | Simuleer de processie zonder bestanden te schrijven. | `False` |
| `--verbose` | Activer een gedetailleerd logbestand voor debuggen. | `False` |
## 📚 Sphinx Integratie

Schaal je documentatie uit tot professionele niveaus met automatische Sphinx i18n ondersteuning.

Wanneer je met de `--sphinx`-vlag draait, README Rosetta:
1.  **Initialiseert Sphinx:** Zet op een `docs/`-bestandsverbinding als dit niet bestaat.
2.  **Auto-configureert i18n:** Updatet `conf.py` met de benodigde `locale_dirs`- en `gettext`-instellingen.
3.  **Haalt Strings Op:** Loopt `gettext` uit om alle translatable strings in je documentatie te vinden.
4.  **Vertaalt PO Bestanden:** Gebruikt de LLM om `.po`-bestanden te vertalen, met behoud van Sphinx-specific syntax zoals `:role:` of `.. directive::`.
5.  **Bouwt HTML Op**: Maakt automatisch lokale HTML-builds voor elk doeltaal.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---
## 📖 GitBook Support

Eenzelschoon onderhoud je een meertalig GitBook.

De `--gitbook` vlag genereert een `SUMMARY.md` bestand dat jouw vertaalde READMEs in een structuur omvormt die compatibel is met de navigatie van GitBook.

- **Automatisch Linken:** Verbindt je Inleiding met je hoofdREADME en maakt lijstitems voor elke vertaalde versie.
- **Taalnamen:** Automatisch resolveert taalcodes (zoals `es`) in hun volledige namen (zoals `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---
## 

Bespare tijd door je projectstandaarden te definieren in `pyproject.toml`:

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```
⚠️ Troubleshootings en Beperkingen

Automatiseerde vertaling met LLM's is krachtig maar kan soms formatingsfouten introduceren, vooral in complexe Sphinx/RST omgevingen.
### Common Problemen
- **Onverhuisbare Sterkjes:** LLMs kunnen moeite hebben om een `` ` `` or ` `` ` `te sluiten.
- **Schriftkopkleuren:** Als een LLM bolding (`**`) toevoegt aan een titel, zal de Sphinx onderlijn niet meer overeenkomen met het tekstlengte.
- **Structurale hallucinaties:** De model kan eigen samenvattingen of "helpvolle" codeblokken toevoegen die niet in de bron zijn.
### Bevoegingscript
We bieden een handig script aan om voortdurende en gemakkelijke translatiefouten te identificeren in uw `.po`-bestanden. Als een vertaling worden geklaarde, zal Sphinx gewoon de oorspronkelijke Engelse tekst gebruiken voor dat specifieke stuk.

```bash
# Run the cleanup utility
python3 scripts/cleanup_translations.py
```

*Opmerking: altijd uw documentatie bouwen. Hoewel Rosetta perfectie zoekt, kan manual correcties van lokale `.po`-bestanden soms nodig zijn voor gevoelige documentatie.*
## 📜 Licentie

Dit project is geleverd onder de MIT-Licentie - zie het [LICENSE](LICENSE)-bestand voor meer informatie.
