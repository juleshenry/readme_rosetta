# 🗿 README Rosetta

**README Rosetta** er en kraftfull automatiseringsverktøy som designs til å oversette din dokumentasjon til flere språk ved hjelp av lokale LLMs via [Ollama](https://ollama.ai/). Det sikrer at ditt prosjekt er tilgjengelig for et globalt publikum while det beholder perfekt Markdown-formatting og dokumentstruktur.

---

## 🌍 README Omsetning

README Rosetta spesialiserer seg på å gjøre din GitHub-prosjekt internasjonell med minimal påkostnad.

- **Multispråksstøtte:** Oversett `README.md` til dozen språk samtidig.
- **Navigasjonstabell:** Automatisk legger inn en navigasjons "stein" (tabell) på toppen av din README, som gjør det lettere for brukerne å switch mellom språkene.
- **Flexibelle modi:**
    - **Splittet modus (Fremtidig standard):** Genererer separate filer (som `README.es.md`, `README.fr.md`) for en ren prosjektdirektorier.
    - **Sammenligged modus (`--no-split`):** Legger alle oversettelser til hovedfilen `README.md`, skilt fra hverandre med HTML-kommentarer.
- **Markdown-bevaring:** Kan håndtere navne-, liste- og koden blokkes for å sikre at oversette utgaven ble funksjonal og godt formatert.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---

## 🛠 Kommando linje (CLI)

KLIs er designet til å være intuitivt likevel mykkelige å bruke.

### Installasjon

```bash
pip install readme-rosetta
```

*Note: Krver at [Ollama](https://ollama.ai/) skal installationeres og kjøres på din maskin.*

### Globale alternativ:

| Alternativ | Beskrivelse | Falsk |
| :--- | :--- | :--- |
| `path` | Veien til kilder eller prosjekt directory. | `README.md` |
| `--langs` | Liste over mål språk-koder (f.eks. `es fr de`). | `[]` |
| `--src-lang` | Kilder-språkskode. | `en` |
| `--model` | Ollama-modell ID å bruke. | `llama3.2` |
| `--readme` | Veien til den hovede utgaven README-filen. | `README.md` |
| `--no-split` | Legger oversettelser til en enkelt fil. | `False` |
| `--dry-run` | Simuler prosessen uten å skrive filer. | `False` |
| `--verbose` | Aktivere detaljert logg for debugging. | `False` |

---

## 📚 Sphinx-integrering

Skala ditt dokumentasjon til profesjonell niva med automatisert Sphinx i18n-støtte.

Når du kjører med `--sphinx` flagget:
1.  **Initialiserer Sphinx:** Installerer en `docs/` directory hvis den ikke eksisterer.
2.  **Auto-konfigurerer i18n:** Oppdaterer `conf.py` med nødvendige `locale_dirs` og `gettext` innstillingene.
3.  **Finner string-streng:** Kjør `gettext` for å finne alle oversettelige strenger i ditt dokumentasjon.
4.  **Oversetter PO-filer:** Bruger LLM til å oversette `.po`-filer, med bevaring av Sphinx-specifikke syntax som `:role:` eller `.. directive::`.
5.  **Bygger HTML:** Automatisk genererer lokaliserede HTML-bygder for hvert mål språk.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---

## 📖 GitBook-støtte

Enkel tilgang til multispråk utgave av din GitBook.

Den `--gitbook` flaggen genererer en `SUMMARY.md` fil som kartes over oversatte READMEs i en struktur som er kompatibel med GitBooks navigasjon.

- **Automatisk linking:** Legger inn introduksjonen til ditt hoved README og skaper liste-item for hvert oversatt versjoner.
- **Språknavn:** Automatisert resolver språk-koder (som `es`) til deres fullte navn (som `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---

## ⚙️ Konfigurering

Reduser tidsplassen ved å definere dine prosjektdefineringer i `pyproject.toml`:

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```
