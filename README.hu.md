# 🗿 README Rosetta

**README Rosetta** egy erősekő eszköz, amely automatizálja dokumentumait több nyelvbe fordítására a helyi LLM-k segítségével [Ollama](https://ollama.ai/) segítségével. Hitelesíti a projektet az egész világgazdagság érdekében, míg megőrzi a perfekt Markdown formátumot és dokumentum szerkezetét.
## 🌍 README Fordulat

Fordulat Rosetta a te GitHub projektjának nemzetközi értékét valódi minimmelényével biztosítja.

- **Több nyelvű támogatás:** Az `README.md` négyzethez százesztílusban dozzsd meg több tucat nyelve között.
- **Navigációs asztal:** Automatikusan a README értéke előtt egy kivágott "köve" (asztal) létesíteni tudja, lehetőség van számos nyelvre kivágásra.
- **Változó módok:
    - **Rövidített modus (Általános):** Széleskörű tüntető fájlokat (pl. `README.es.md`, `README.fr.md`) generál és ezzel a tisztában maradsz.
    - **Egyszeri modus (`--no-split`):** A fordításokat az egyes fájlokra hozzáadja a `README.md` fájlba, külön HTML-kommentekkel elkülöníteni tudja.
- **Markdowntartalom:** Szakosan figyelembe veszi a fejezeteket, listákat és kód blokkokat, hogy a fordított kimenet függőleges maradjon és működjön.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---
## ⚙️ Állomány szabályos felületű interfésze (CLI)

Az interfész intuitívan és erőszakosan is tervezett.
### Telepítés
```bash
pip install readme-rosetta
```

*Megjegyzés: Ollama rendszerét kell telepíteni és indítani a számítógépen.* 

    [Ollama](https://ollama.ai/)
### Globális Beállítások

| Opció | Leírás | Általános |
| :--- | :--- | :--- |
| `path` | Forrás file útvonala vagy projekt mappájának útvonala. | `README.md` |
| `--langs` | Cél nyelvkódok listája (például `es fr de`). | `[]` |
| `--src-lang` | Forrás nyelkód. | `en` |
| `--model` | Ollama modell ID, amelyet használni kell. | `llama3.2` |
| `--readme` | Az útvonal az üdvösségi fájl fő README fileje. | `README.md` |
| `--no-split` | Fordításokat egyetlen fájlba juttatni. | `False` |
| `--dry-run` | Folyamat simulálása anélkül, hogy fájlokat írja le. | `False` |
| `--verbose` | Bonyolult jogutak engedélyezése a debuggeléshez. | `False` |
## 📚 Sphinx Integrációs Elszerelés

A professzionális szintekre kiterjesztett dokumentáció automatikus Sphinx i18n támogatásával lépjen be.

Amikor a `--sphinx` zást isznak, README Rosetta:
1.  **Beállítja a Sphinx-et:** Hozza létre egy `docs/` iránytort amely nem létezik.
2.  **Automatikusan konfigurálja az i18n-t:** Frissíti a `conf.py`-et a szükséges `locale_dirs` és `gettext` beállításokkal.
3.  **Kivonja Az Sztringeket:** A `gettext` segítségével keresi meg az összes forditható sztringet a dokumentációban.
4.  **Fordítja PO Fájlokat:** Az LLM segítségével fordítja a `.po` fájlokat, úgyhogy megtartja a Sphinx-specifikus szintaxist mint például az `:role:` vagy az `.. directive::`.
5.  **Gyűjti Ki Az HTML-t:** Automatikusan létrehozza a minden céltarget nyelv területén megvalósított helyi HTML buildet.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---
## 📖 GitBook Support

Leányítási módszerrel egyszerűen karbantartható egy több nyelvű GitBook.

A `--gitbook` zászló megenerálja a `SUMMARY.md` fájlt, amely a fordított README-eket a strukturával hozza ki egybe a navigációval rendelkező GitBook működésének kompatibilitásába.

- **Automatikus Hozzászámlálás:** A bevezetőt a fő README-höz köti le, és listai elemeket létesít every fordított verzióra.
- **Nyelvnevek:** Automatikusan megoldja a nyelkkódok (mint `es`) az egész nevüket (mint `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---
## ⚙️ Konfiguráció

A projektekhez szabályosabb átmenettel meghatározhatja a saját előfőbbet `pyproject.toml`:

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
⚠️ Aviskodás & Korlátozások

A számítógépes lefordítás LLM-kkal erősek, de az esetleges formátumhiányokat, különösen a komplex Sphinx/RST környezetben szintén megjeleníthetik.
### Közös Hibák
- **Ellenálló Visszaítéletek:** Az LLMeknek súlyos lehetőségük van arra, hogy be nem zárják  `` `` `` or `` `` `` ` stringot.
- **Cím Hosszúságok:** Ha az LLMs hozzáadnak bontás ( ``**`` ) egy címhez, a Sphinx aláhúzás a többi szövegetől nem érhet fel.
- **Szerkezeti Kitalációk:** A modellek esetleg saját összefoglalókat vagy segítő code blockokat hozzanak létre, amelyek nincsenek a forráskódban.
### Tisztító szkript
Másolatot adjunk egy olyan felülvizsgálati eszközt, amely az általános fordítási hibákat az `.po` fájloknak tartalmazza. Ha egy fordítás tisztítható, a Sphinx egyszerűen visszafordul az eredeti angol szövegre annak helyére.

```bash
# Run the cleanup utility
python3 scripts/cleanup_translations.py
```

*Nota bene: Mindig ellenőrizze a dokumentációs építkezéseit. Míg Rosetta célt tirt a tökéletesség, akkor is van esélye az olyan hibák átvizsgálására, hogy a helyi `.po` fájlokban.
## 📜 Lícencék

Ez a projekta az MIT Licencs alapján van engedélyezve – lásd a [LICENSE](LICENSE) fájlt azonban részletekért.
