# 🗿 README Rosetta

**README Rosetta** er et stort automatiseringsverktøj, der tilbyder at oversætte dine dokumentation til mange sprog ved hjælp af lokale LLMs via [Ollama](https://ollama.ai/). Det sikrer, at din projekt er tilgængelig for en global befolkning mens man beholder perfekt Markdown-indhold og dokumentstruktur.
## 🌍 README Oversættelse

README Rosetta specialiserer sig i at gøre din GitHub-projekt international med minimale kræfter.

- **Multi-lingvisme:** Oversæt din `README.md` til årene og samtidigt.
- **Navigations-tabel:** Automatisk indsætter en navigations "sten" (tabel) på toppen af din README, som giver brugeren hurtig adgang til at skifte mellem sprogene.
- **Flexiblene Mode:
    - **Split Mode (Fremgangsrig):** Genererer separate filer (f.eks. `README.es.md`, `README.fr.md`) for en rene projektstruktur.
    - **Unified Mode (`--no-split`):** Tilføjer alle oversættelser til den hovede `README.md`-filen, adskilt af HTML-kommentarer.
- **Marknadspræsentation:** Intelligent håndtering af hovedlinjer, liste- og code-blocks for at sikre, at oversættelsen bliver funktionsdugt og bienformet.
## 💻 Kommandolinje-Brugergrænsegang (Kommando Linje-Brugérgrænsegang)

Kommandolinjen er designed at være intuitivt og kraftfuld.
### Installation

```bash
pip install readme-rosetta
```

*Nota: Kræver [Ollama](https://ollama.ai/) at at være installeret og kørt på din computer.*
### Globale Indstillinger

| Opdagelse | Beskrivelse | Forkastet |
| :--- | :--- | :--- |
| `path` | Vej til kilder eller projektionsafdeling. | `README.md` |
| `--langs` | Liste over målindstillinger (f.eks. `es fr de`). | `[]` |
| `--src-lang` | Kilde-sprogkode. | `en` |
| `--model` | Ollama-Model ID at bruge. | `llama3.2` |
| `--readme` | Vej til hovedudgivernes README-fil. | `README.md` |
| `--no-split` | Tilføj oversættelser til et enkelt fil. | `False` |
| `--dry-run` | Simuler proces uden at skrive filer. | `False` |
| `--verbose` | Aktiver detaljeret log for udvidelse. | `False` |
## 📚 Sphinx Integration

Skal din dokumentation til professionelle niveauer med automatisk Sphinx i18n-støtte.

Når du kører med `--sphinx`-flaggen, README Rosetta:
1.  **Iniserer Sphinx:** Indsætter en `docs/`-kaot efter behovet.
2.  **Auto-konfigurerer i18n:** Opdaterer `conf.py` med nødvendige `locale_dirs` og `gettext`-indstillingerne.
3.  **Tager Strænger:** Gennemfører `gettext` for at finde alle oversættelige strænger i din dokumentation.
4.  **Oversætter PO Files:** Brugere LLM til at oversætte `.po`-filer, hviligt bevarer Sphinx-specifik syntax som `:role:` eller `.. directive::`.
5.  **Bygger HTML:** Genererer automatisk lokaliseret HTML-bygning for hvert målmedvirkende sprog.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---
## 📖 GitBook Support

Elteligent underhåll en multi-tilsprogende GitBook.

Den `--gitbook` flag genererer en `SUMMARY.md` fil, der mapper din oversat README til en struktur, som er kompatibel med GitBooks navigation.

- **Automatisk Linking:** Ligner introduktionen til din hoved README og skaber items i listen for every oversat version.
- **Tilbagefærdig Navne:** Automatisk resolver talekoder (som `es`) til deres fulde navne (som `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---
## ⚙️ Konfiguration

Reduzér tid ved at definere dit projektstandardindstillinger i `pyproject.toml`:

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
## ⚠️ Fejlfinding og Begrensninger

Automatiseret oversættelse med LLM's er stærk men kan i nogle tilfælde introducere formattingsskader, især i komplexe Sphinx/RST-miljøer.
### Fælleproblemer
- **Uenligte tegnmeder:** LLMs kan fejlåde at lukke en `` ` `` or ` `` ` `streng.
- **Titel-længde:** Hvis et LLM tilføjer bolder (`**`) til en titel, Sphinx understrening kan ikke længere være i tråd med tekstens længde.
- **Struktureret hallucinationer:** Modellen kan prøve at tilføje eget sammanfattende blok eller "hjælpsomme" kodeblokke som ikke er i kildekoden.
### RenetningsSkript
Vi tilbyder en tilvirkningsskript til at identificere og skåne fælles fejl i dine `.po`-filer. Hvis en oversættelse kleares, vil Sphinx blot faller tilbage til oprindelig engelsk tekst for den enkelte string.

```bash
# Run the cleanup utility
python3 scripts/cleanup_translations.py
```

*Notat: Sågendebygge dokumentation bygger aldrig på perfektion, selv om Rosetta stræber efter det. Manuelt korrektion af lokaliseret `.po`-fil er ofte nødvendigt for høje-stakes dokumentation.*
## 📜 Licensen

Dette projekt er licensieret under den MIT Licens - se [LICENSE](LICENSE) filen for detaljer.
