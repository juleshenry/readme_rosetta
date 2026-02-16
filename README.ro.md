# 🗿 README Rosetta

**Înțelegere despre Rosetta**

**Înțelegere despre Rosetta** este un instrument puternic de automatizare proiectată pentru traducerea documentațiilor dvs. în diverse limbi folosite utilizând local LLMs printr-unul din [Ollama](https://ollama.ai/). Asigură astfel ca proiectul dumneavoastră să fie accesibil unui public global pe timpul maintainarea perfectă a formatului Markdown și a structurii documentare.

---
## 🌍 Traducerea README

README Rosetta se specializează în facutul cațiunii dvs. projectului GitHub internațional cu minimă efort.

- **Suport multilingv:** Traduce `README.md` în zeci de limbi pentru a face dumneavoastră șansa să vă programezeți pe piață globală.
- **Tabloul navigație:** Prendă automat un "piatră" (tablou) de navigație la începutul dvs. README, care vă permite să schimbați ușor între limbi.
- **Modele flexibile:**
    - **Modul Split (Stâncă Default):** Generează fișiere separate (de exemplu, `README.es.md`, `README.fr.md`) pentru o structură de proiect netedă.
    - **Modul Unitat (`--no-split`):** Adauga toate traduceri în fișiera principală `README.md`, separate prin comentarii HTML.
- **Prieteneța formatului Markdown:** Manipulează inteligent capacele de headers, liste și blocuri de cod pentru a menține outputul tradus functional și formatat.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---
## 💻 Interfața liniei de comand (CLI)

Interfața cliului este proiectată pentru a fi intuitivă și puternică.
### Instalare

```bash
pip install readme-rosetta
```

*Notație: Rezolvă o problemă de instalare și asigurați-vă că [Ollama](https://ollama.ai/) este instalat și înainte să fie rulat.
### Opțiuni globale

| Opțiune | Descriere | Default |
| :--- | :--- | :--- |
| `path` | Locația fișierului sau directorului sursă. | `README.md` |
| `--langs` | Lista codurilor de limbă țintă (de exemplu, `es fr de`). | `[]` |
| `--src-lang` | Codul limbi sursă. | `en` |
| `--model` | ID-ul modelului Ollama pentru utilizare. | `llama3.2` |
| `--readme` | Locația fișierului principal de descriere a output-ului README. | `README.md` |
| `--no-split` | Trimite traducerile într-un singur fișier. | `False` |
| `--dry-run` | Simulează procesul fără să scrie fișiere. | `False` |
| `--verbose` | Activează logul detaliat pentru depistarea greșeli. | `False` |
## 📚 Integrare cu Sphinx

Aşeză documentaţia dvs. la nivel profesional cu sprijin automatizat de Sphinx i18n.

În modul `--sphinx`, README Rosetta:
1.  **Întreprinde configurarea iniţială a Sphinx:** Crează o `docs/` directorie dacă nu există.
2.  **Auto-configura ţinta i18n:** Actualizează `conf.py` cu setările necesare `locale_dirs` și `gettext`.
3.  **Extractează Stringuri:** Răspunde la comanda `gettext` pentru a găsi toate stringurile translatabile în documentaţia dvs.
4.  **Traduce fişiere PO:** Utilizează LLM pentru a traduce fişierele `.po`, preservând sintaxa Sphinx-specifică cum ar fi `:role:` sau `.. directive::`.
5.  **Creează HTML Localizate:** Generează automat construcţii HTML localizate pentru fiecare limbă target.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---
## 📖 Suportul pentru GitBook

Cu ușurință, puteți menține unGitBook multilingv.

Linzile `--gitbook` generă o linie de cod `SUMMARY.md` care mapping-ul dvs. traducerilor README într-un structură compatibilă cu navigația GitBook.

- **Automatizarea Linkării:** Înlegă linkurile intrării în mainul dvs. README și creează puncte listă pentru fiecare versiune tradusă.
- **Numele Limbilor:** Resolva automat codurile limbilor (ca `es`) în numele lor completi (ca `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---
## ⚙️ Confiгurație

Salvează timp prin definirea standardelor projectuale în `pyproject.toml`:

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```
### Probleme de soluționare și limitări 

Traducerea automată cu ajutorul unor modeli de inteligentă artificială este puternică dar poate introduce artefacte de formatare într-unii medii Sphinx/RST complexe.
### Probleme Comune
- **Așezări Înșiruite cu Acreșuni:** LLMs pot eftimiza o `` ` `` or ` `` ` ` string nu se închide corect.
- **Longități de Titluri:** Dacă un model LLM adaugă semnificații (`**`) la o titlu, underline-ul Sphinx poate mai puțin rămâne în concordanță cu lungimea textului.
- **Hallucinații Structurale:** Modelul poate încerca să adauge propiile sumari sau blocaje de cod care nu sunt preșente în sursa originală.
### Scriptul de limpeză
Pentru a vă furniza un script utilitate pentru identificarea și eliminarea greșelilor comune ale traducerii în fișierele dictionarului `.po`. Dacă o traducere este eliminată, Sphinx va rămâne cu textul original englez pentru acea frază.

```bash
# Run the cleanup utility
python3 scripts/cleanup_translations.py
```

*Nota: În mod deosebit, revedeți întotdeaun construcțiile documentare. Deoarece Rosetta vă aspiră la perfecție, corectarea manuană a fișierelor localizate ale dictionarului `.po` este deseori nevoie pentru documentații critice.*
## 📜 Licență

Acest proiect este licențiat sub licența MIT - văzitți fișierul [LICENSE](LICENSE) pentru detalii.
