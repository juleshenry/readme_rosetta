# 🗿 README Rosetta

**README Roseta** este o instrumentă de automatizare puternică proiectată pentru a traduce documentația dvs. în multiple limbi folosind locali LLMs prin [Ollama](https://ollama.ai/). Acest lucru asigură ca proiectul dvs. să fie accesibil publicului global cu perfecta menținere a formatului Markdown și structura documentară.

---

## 🌍 Traducerea README

ROSETTA_Roseta se specializează în făcându-vă proiectul GitHub internațional cu minimal efort.

- **Suport la multe limbi:** Traduceți `README.md` în doze de limbi diferite simultan.
- **Tabelul de navigare:** Autogenerarea unui "piscuri" (tabel) de navigație la începutul dvs. README, care permite utilizatorilor să schimbe ușor între limbi.
- **Moduri flexibile:
    - **Modul Split (Nemodificat):** Găsește separate file (de exemplu `README.es.md`, `README.fr.md`) pentru o structură proiect a curatat.
    - **Modul Unificat (`--no-split`):** Adaugă toate traducerile în fișiera principală `README.md`, separat prin comentarii HTML.
- **Menținerea formatului Markdown:** Manipulează înțelegerea cu capetele, listele și blocurile cod pentru a asigura ca outputul tradus să rămână funcțional și de format foarte bun.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---

## 🛠 Interfața de Comandă (CLI)

Interfața de comandă este proiectată pentru a fi inteligentă și puternică.

### Instalare

```bash
pip install readme-rosetta
```

*Nota: Requiere [Ollama](https://ollama.ai/) să fie instalat și rulat pe sistemul dvs.*

### Opțiuni globale

| Opțiune | Descriere | Valoarea standard |
| :--- | :--- | :--- |
| `path` | Drumul la fișiera sursă sau directorul proiect. | `README.md` |
| `--langs` | Lista codurilor de limbă țintă (de exemplu `es fr de`). | `[]` |
| `--src-lang` | Codul limbii sursă. | `en` |
| `--model` | ID-ul modelului Ollama să folosiți. | `llama3.2` |
| `--readme` | Drumul la fișiera principală a output-ului README. | `README.md` |
| `--no-split` | Adaugarea traducerilor într-o singură fișieră. | `False` |
| `--dry-run` | Simularea procesului fără scriere a fișierelor. | `False` |
| `--verbose` | Activarea logaringii detaliate pentru depistarea de greșeli. | `False` |

---

## 📚 Integrare cu Sphinx

Același nivel profesional de documentare, automatizată cu suport la i18n pentru Sphinx.

În momentul în care rulați cu flag-ul `--sphinx`, README Roseta:
1.  **Initiază Sphinx:** Înființează o `docs/` directorie dacă nu există.
2.  **Auto-configura i18n:** Actualizează `conf.py` cu informațiile necesare `locale_dirs` și `gettext`.
3.  **Extrage fișierele PO:** Rula `gettext` pentru a găsi toate stringurile traducibile în documentația dvs.
4.  **Traduce fișierele PO:** Folosește LLM-ul pentru a traduce `.po` fișiere, menținând sintaxa specifică Sphinxului cum `:role:` sau `.. directive::`.
5.  **Generează HTML:** Crea localizate HTML personalizate pentru fiecare limbă dvs.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---

## 📖 Suport la GitBook

Ești în stare să menții o versiune multilingual a GitBook.

Flag-ul `--gitbook` generează un fișier `SUMMARY.md` care mapează versiunile traduse ale dvs. README cu o structură compatibilă cu navigația GitBook.

- **Legarea la introducție:** Legați Introduction-ul la main-ul dvs. README și creați lista de iteme pentru fiecare traducere.
- **Numele limbi:** Resolva codurile limbilor (de exemplu `es`) în numele lor completi (de exemplu `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---

## ⚙️ Confișurații

Salvați timpul dvs. prin definirea standardelor proiectului.

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

## 📜 Licență

Acest proiect este lăsat sub licența MIT - vezi fișiera [LICENSE](LICENSE) pentru detalii.
