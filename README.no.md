# 🗿 README Rosetta

**README Rosetta** er en kraftfull automatiseringstool som designes for å oversette dine dokumenter til flere språk ved hjelp av lokale LLMs via [Ollama](https://ollama.ai/). Det garanterer at ditt prosjekt er tilgjengelig for en global audiens mens det beholdt perfekt Markdown-formatting og dokumentstruktur.
## 🌍 README Oversettelse

README Rosetta tilbyr at gjøre din GitHub-prosjekt internasjonal med minimal innsats.

- **Mannlig support for flere språk:** Oversett `README.md` til flere tiener samtidig.
- **Navigasjonsøyra:** Automatisk tillegger en navigasjons "stein" (tabel) på toppen av dine README, så brukspersonerne kan raskt switch mellem språkene.
- **Anpasselige modi:**
    - **Splittet mode (fremgangsmålet):** Genererer separate filer (fra `README.es.md` til `README.fr.md`) for en ren prosjektsstruktur.
    - **Enhetlig mode (`--no-split`):** Legger alle oversettelsene i den hovede `README.md`-filen, separert av HTML-kommentarer.
- **Markdown-bevaring:** Intelligent håndterer hedringer, liste og kodblokk for å sikre at oversatt output er funksjonal og velformet.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```
## ⚙️ Kommandelinje-Interfase (KLI)

KLI er designed til å være intuitiv og kraftfull.
### Installasjon

```bash
pip install readme-rosetta
```

*Notat: Kraver [Ollama](https://ollama.ai/) å være installert og løpende på din system.
### Global Options

| Option | Beskrivelse | Standardverdi |
| :--- | :--- | :--- |
| `path` | Vei til kildefil eller prosjekt-utfolder. | `README.md` |
| `--langs` | Liste over målspråk-koder (f.eks. `es fr de`). | `[]` |
| `--src-lang` | Kilderingskode for kildespråket. | `en` |
| `--model` | Ollama-modell ID å bruke. | `llama3.2` |
| `--readme` | Veien til hovedutgivelses README-filen. | `README.md` |
| `--no-split` | Legg oversettelse til en enkelt fil. | `False` |
| `--dry-run` | Simuler prosessen uten å skrive filer. | `False` |
| `--verbose` | Aktivere detaljert logg for debugging. | `False` |
## 📚 Sphinx Integrering

Skal du din dokumentasjon til professionelle nivå med automatisert Sphinx i18n-støtte.

Når du kører med `--sphinx`-bra, README Rosetta:
1.  **Initierer Sphinx:** Installerer et `docs/`-ordinner om det ikke eksisterer.
2.  **Auto-konfigurerer i18n:** Oppdaterer `conf.py` med nødvendige `locale_dirs` og `gettext` innstillinger.
3.  **Trekker ut stringer:** Kører `gettext` for å finne alle oversettlelige stringer i din dokumentasjon.
4.  **Oversetter PO-filer:** Bruker LLM til å oversette `.po`-filer, med presisjon for Sphinx-espesialt syntax som `:role:` eller `.. directive::`.
5.  **Bygger HTML:** Genererer automatisk lokaliseret HTML-bygg for hvert målmedlet.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```
## 📖 GitBook Support

Enkelt vedlikeholde et multi- språkGitBook.

Den `--gitbook` flagen generer en `SUMMARY.md` fil som kartlegger dine oversatte READMEs i en struktur bestemt av GitBooks navigasjon.

- **Automatisk Linking:** Ligner inn introduksjonen til din hoved README og skaper listebitatter for hver oversatt versjon.
- **Språknavn:** Automatisk løser språkkoder (som `es`) til deres fullte navn (som `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---
## ⚙️ Konfigurasjon

Reducer tid ved å definere ditt prosjektstandardverdier i `pyproject.toml`:

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```
## ⚠️ Feilkjennings- og begrunnelsesområder

Automatisert oversettelse med LLMs er sterk, men kan noen ganger føre til formatjonsskader, særdeles i komplexe Sphinx/RST-områder.
### Utbredte Problemer
- **Uenligt tillegg av backtick:** LLMs kan feile ved å lukke en `` ` `` or ` `` ` ` streng.
- **Lenker på overskrifter:** Hvis et LLM legger til bokstavsforming (`**`) til en overskrift, Sphinx underlining kan ikke lenger være i enig med tekstlengden.
- **Strukturhallusinasjoner:** Modellen kan prøve å tilføye sine egne sammentog eller "hjelpfulle" koden blokkes som ikke finnes i kildecode.
### Renhelskript
Vi tilbyr en utilitets-skrikt som identifiserer og renner ut vanlige oversettelsesfeil i dine `.po`-filer. Hvis en oversettelse rennes, vil Sphinx bare velge original-engelske tekst for den bestemte string.

```bash
# Run the cleanup utility
python3 scripts/cleanup_translations.py
```

*Nota: Altid sjekk din dokumentbygning. Hvis Rosetta går etter perfeksjon, må man gjøre håndhavs korrigering av lokaliserte `.po`-filer noen ganger for høye-stakes dokumentasjon.*
## 📜 Lisens

Dette prosjektet er licensert under MIT-Lisensen - se [LICENSE](LICENSE) filen for detaljer.
