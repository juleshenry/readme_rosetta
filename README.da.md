# 🗿 README Rosetta

**README Rosetta** er en kraftfuld automatiskt omhændelsestool til at oversætte dit dokumenter til flere sprog ved hjælp af lokale LLMs via [Ollama](https://ollama.ai/). Det sikrer, at din projekt er tilgængelig for en global befolkning med perfekt Markdown-formatsning og dokumentsstruktur.

---

## 🌍 README Oversætning

README Rosetta specialiserer sig i at gøre dit GitHub-projekt international med minimal indsats.

- **Multi-sprogstøtte:** Oversæt dine `README.md` i dussiner af sprog samtidigt.
- **Navigations tabel:** Automatisk tilføjer en navigations "sten" (tabel) til toppen af din README, hvilket giver brugeren mulighed for hurtig overgang mellem sprogene.
- **Flexibele modi:
    - **Splitted Mode (Standard):** Genererer separate filer (f.eks. `README.es.md`, `README.fr.md`) for en rensning projektstruktur.
    - **Sammenlignet Mode (`--no-split`):** Tilføjer alle oversættelser til den hovede `README.md`-fil, skilt af HTML Kommentarer.
- **Markdown-bevarelse:** Intelligent håndterer hovedlinjer, listeindlag og kodeblokke for at sikre, at oversættelsen bliver funktional og godt formatert.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---

## 🛠 Kommandelinemaskine (CLI)

Denne CLI er udformet til at være intuirt og kraftfuld.

### Indstilling

```bash
pip install readme-rosetta
```

*Notaion: Kræver [Ollama](https://ollama.ai/) at være installeret og kørt på din system.*

### Verdensniveau-optioser

| Option | Beskrivelse | Standard |
| :--- | :--- | :--- |
| `path` | Sti til kilderne eller projektets direktør. | `README.md` |
| `--langs` | Liste af target language codes (f.eks. `es fr de`). | `[]` |
| `--src-lang` | Kilder-sprogkode. | `en` |
| `--model` | Ollama modell ID at bruge. | `llama3.2` |
| `--readme` | Sti til den hovede output README-fil. | `README.md` |
| `--no-split` | Tilføje oversættelser i en enkelt fil. | `False` |
| `--dry-run` | Simuler processen uden at skrive filer. | `False` |
| `--verbose` | Aktivere detaljeret logging for udviklingsformål. | `False` |

---

## 📚 Sphinx-integration

Skal din dokumentation op i professionelle niveauer med automatiseret Sphinx i18n-støtte.

Når du kører med `--sphinx`-flaggen, README Rosetta:
1.  **Initaliserer Sphinx:** Opstiller en `docs/`-yrkede directory hvis det ikke eksisterer.
2.  **Auto-konfigurerer i18n:** Opdaterer `conf.py` med `locale_dirs` og `gettext`-indsætninger.
3.  **Trækker ud fra strings:** Kører `gettext` til at finde alle oversættelige strings i din dokumentation.
4.  **Oversæt PO-filer:** Bruger LLM'et til at oversætte `.po`-filer, med bevidsthed om Sphinx-specifikke syntax som `:role:` eller `.. directive::`.
5.  **Bygger HTML:** Automatisk genererer lokaliserede HTML-bygninger for every target language.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---

## 📖 GitBook-støtte

Easily-mætig til at vedligeholde et multi-lingvigt GitBook.

Den `--gitbook`-flagg generer en `SUMMARY.md`-fil, der tildeles din oversættede READMEs i en struktur, der er kompatibel med GitBooks navigering.

- **Automatisk länkning:** Knapper introduktionen til den hovede README og opretter listeindlag for hver oversættelse.
- **Sprognavne:** Automatisk resolver sprogkode (som `es`) til deres fulde navne (som `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---

## ⚙️ Konfiguration

Reducer tid og indsats ved at definere dine projekt- standardindstillinger i `pyproject.toml`:

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```
