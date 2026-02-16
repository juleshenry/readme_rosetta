# 🗿 README Rosetta

**README Rosetta** a hatékony automatizálási eszköz, amely számos nyelvre lefordítja dokumentációt helyi LLM-k segítségével a [Ollama](https://ollama.ai/) segítségével. Ezzel biztosítja projektjét a globális közönség számára, míg az eredeti Markdown formátumot és dokumentációs szerkezetet megőrzi.

---

## 🌍 README Fordítás

A README Rosetta a GitHub projectnek nemzeti elérhetőséget biztosítja minimal erőfeszítéssel.

*   **Többnyelvű támogatás:** Az `README.md` -et több száz nyelvre fordítsa egyetlen lépésben.
*   **Navigációs tabla:** Automa-tikusan a navigációs "kő" (tábla) elrendezése a README tetején, lehetővé téve a felhasználóknak gyors csatolásokat nyílásban.
*   **Módosított működési modelljei:**
    *   **Szétválasztott módd (Állapotos):** Szétszorozza a fájlakat (pl. `README.es.md`, `README.fr.md`), hogy tiszta projektstruktúrát biztosítja.
    *   **Egyesített módd (`--no-split`):** A fordított szöveget az eredeti `README.md` -be teszi, HTML- kommentekkel különíti el a fájlakat.
*   **Markdown megőrzése:** Intelligens kezelésével a fejbetűk, listák és kód blokkokat garantálja, hogy a fordított eredmény funkcionál és jól formált maradjon.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---

## 🛠 Osztályozott felület (CLI)

A CLI intuitív, de erősek.

### Beállítás

```bash
pip install readme-rosetta
```

*   **Note:** A [Ollama](https://ollama.ai/) kell beállítva és futtatva a rendszeren.

### Globális opciók

| Opción | Leírás | Álapértelmezett |
| :--- | :--- | :--- |
| `path` | Forrási fájl vagy projekt irányzata. | `README.md` |
| `--langs` | Cél nyelvek listája (pl. `es fr de`). | `[]` |
| `--src-lang` | Forrás nyelve kódja. | `en` |
| `--model` | Ollama model ID. | `llama3.2` |
| `--readme` | A fő kiadott README fájl irányzata. | `README.md` |
| `--no-split` | Fordítások hozzáadása egyetlen fájlhoz. | `False` |
| `--dry-run` | A folyamat simulálása a fileírás nélkül. | `False` |
| `--verbose` | Rögzítési részletek gyorsítása a fejlesztők számára. | `False` |

---

## 📚 Sphinx Integráció

A dokumentáció szintén szakemberek számára profi színvonalon lép ki automatizált Sphinx i18n támogatással.

Ha a rosetta_cl.py -parancsot használja, az README Rosetta:
1.  **Sphinx inicializálása:** Rendezettséget teremt a `docs/` kiterjesztésű irányzaton, ha nincsen.
2.  **Auto-figyelem i18n:** Frissíti a `conf.py` -vel az egyes `locale_dirs` és `gettext` beállításokat.
3.  **Szövegrészletek kivonása:** Rögzített szöveget keres a dokumentációban, hogy fordítsa a `gettext` -val.
4.  **PO fájlok fordítása:** A LLM-t használva a fordított `.po` -fájlt tartalmazó PO fájlok megtartja a Sphinx-specifikus szintaxis, mint például a `:role:` vagy `.. directive::`.
5.  **HTML létrehozás:** Automatikusan generálja az összes cél nyelvekhez kielégítő HTML építményt.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---

## 📖 GitBook Támogatás

A dokumentáció szintén könnyen támogatható GitBook használattal.

Az `--gitbook` -parancs generál egy `SUMMARY.md` fájlt, amely a fordított README-kat az úgynevezett "fordított átjáró" strukturába helyezi, hogy a GitBook navigációs funkcióit is megtekinthesse.

*   **Automatikus összefüggés:** A bevezetőbe a fő README és a különböző fordított változatok között összekapcsolja az úgynevezett "fordított lista" (list item) segítségével.
*   **Nyelvek neve:** Automatikusan meghatározza a nyelvek kódjának két szabását, mint például `es`.

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---

## ⚙️ Beállítások

A projekt default beállításait az `pyproject.toml` -fájlba rögzíthetők.

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```
