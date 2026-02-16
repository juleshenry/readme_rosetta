# 🗿 README Rosetta

**README Rosetta** är en mäktig automatiseringsverktyg som designas för att översätta din dokumentation till flera språk med lokala LLMs via [Ollama](https://ollama.ai/). Det garanterar ditt projekt är tillgängligt för en global publik med perfekt Markdown-format och dokumentstruktur.

---
## 🌍 README Översättning

README Rosetta specialiserar på att göra ditt GitHub-projekt internationellt med minimal ansträngning.

- **Målförspråkssupport:** Översätter din `README.md` till dusin av språk samtidigt.
- **Navigationsbordet:** Automatiskt försätter ett navigerings "sten" (tabell) på toppen av ditt README, vilket gör det möjligt för användaren att snabbt byta mellan språk.
- **Anpassningsbara läge:
    - **Splitteda läget (Förråd):** Genererar separat filer (t.ex. `README.es.md`, `README.fr.md`) för en rengörd projektstruktur.
    - **Sammanlagt läge (`--no-split`):** Försätter alla översättningar till den huvudsakliga `README.md` filen, skiljda av HTML-kommentarer.
- **Markdown-behandling:** Intelligentt hanterar rubrikerna, listorna och kodblocken för att säkerställa att den översatta utgången blir funktionsduglig och välformad.
## ⚙️ Kommandradgränssnitt (KRAV)

KRAV är designat för att vara intuitivt och kraftfullt.
### Installation

```bash
pip install readme-rosetta
```

*Notek: Kräver [Ollama](https://ollama.ai/) att installerats och körs på din dator.*
### Global Options

| Övrigt | Beskrivning | Förespråk |
| :--- | :--- | :--- |
| `path` | Väg till källfil eller projektupplösning. | `README.md` |
| `--langs` | Lista av mål Språk koderna (f.eks. `es fr de`). | `[]` |
| `--src-lang` | Källspråket kod. | `en` |
| `--model` | Ollama modell ID att använda. | `llama3.2` |
| `--readme` | Väg till huvudutgåva README filen. | `README.md` |
| `--no-split` | Lägg översättningar i ett enda fil. | `False` |
| `--dry-run` | Simulera processen utan att skriva filer. | `False` |
| `--verbose` | Aktivera detaljerad loggning för utvärdering. | `False` |
## 📚 Integration med Sphinx

Skala dina dokument för professionella nivåer med automatiskt integrering av Sphinx i18n-stöd.

När du körs med `--sphinx`-fanningsmärket README Rosetta:
1.  **Initialisera Sphinx:** Skapar en `docs/`-yrke om den inte existerar.
2.  **Automatiserar i18n:** Uppdaterar `conf.py` med de nödvändiga `locale_dirs` och `gettext` inställningarna.
3.  **Utträffar Strängar:** Kör `gettext` för att hitta alla överskrivbara strängar i ditt dokument.
4.  **Översätter PO-filer:** Använder LLM för att översätta `.po`-filer, med uppmärksamhet på Sphinx-specifika syntax som `:role:` eller `.. directive::`.
5.  **Skapar HTML-byggen:** Skapar automatiskt lokaliserade HTML-byggen för varje målmedborgarspråk.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---
## 📖 GitBook Support

Enkelt upprätta en multi-språkig GitBook.

Den `--gitbook` flaggen genererar en `SUMMARY.md` fil som mappar dina översatta READMEs till en struktur som är kompatibel med GitBooks navigering.

* **Automatisk Länkning:** Länkar inledningen till din huvud README och skapar list-objekt för varje översatt version.
* **Språks namn:** Automatiskt löser språkcoderna (som `es`) ut till deras fulla namn (som `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---
## ⚙️ Konfiguration

Spara tid genom att definiera ditt projektstandard i `pyproject.toml`:

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```
⚠️ Felsökning och begränsningar

Automatisk översättning med LLMs är kraftfull men kan ibland introducera formatförfalskningar, särskilt i komplexa Sphinx/RST-miljöer.
### Vanliga problem
- **Uenligt citatte:** LLMs kan missa att stänga en `` ` `` or ` `` ` `-string.
- **Radskalan för rubriker:** Om ett LLM lägger till fett (`**`) till en titel, ska Sphinx underlinning kanske inte längre matcha textens längd.
- **Strukturläckage:** Modellen kan försöka lägga till sitt egna sammanfattningar eller "hjälpsamma" kodbloker som inte är i källan.
### Reningskript
Vi förser med en tillverkningsutjämnhet för att identifiera och rensa vanliga översättningsfel i din `.po`-filer. Om en översättning rensas, följer Sphinx simplement det ursprungliga engelska textet för den angivna stringen.

```bash
# Run the cleanup utility
python3 scripts/cleanup_translations.py
```

*Notera alltid att du ska granskning av dina dokumentbyggande. Även om Rosetta måste sträva efter perfektion är manlig korrektur av localiserade `.po`-filer ibland nödvändig för högt ansvariga dokument.*
## 📜 Licens

Denna projekt licensieras under MIT-Licensen - se filen [LICENZ](LICENSE) för detaljer.
