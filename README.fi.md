# 🗿 README Rosetta

**README Rosetta** on tietystyä automaattinen käytölaitteisto, joka kääntää dokumenteja moniin kielihankkeisiin käyttämällä paikallisia LLM:itä [Ollama](https://ollama.ai/). Se yhtenistää projectin globaalia näyttökohdassa vähän työrautena.

---

## 🌍 README Kääntäminen

README Rosetta on erityisen kehitetty GitHub-projektille kansainvälisten yhteiskunnan saataville.

- **Monikielinen tuki:** Kääntä `README.md` kieliin kolikon eri kielellä samanaikaisesti.
- **Navigointitaulu:** Automaattisesti lopettaa table tietoa asettamalla käyttäjille nopeampaa siirtymistä kielillä.
- **Muuntumiskykyiset mallei:** 
    - **Hajautunut mallei (Pohjoismaiset):** luodakseerivat erillisiä tietoaineistoja (kuten `README.es.md`, `README.fr.md`) etenkin kaunean asentamiseksi.
    - **Yhdistetty mallei (`--no-split`):** liittää käännökset `README.md`-tiedostoon, joiden välillä on HTML-kommentteja.
- **Markdown ylläpito:** hallitsevat sähköistä headeria, listaa ja koodipaloja tietokäyttöön ottaakseen.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---

## 🛠 Käsikäyttäjänvalikoima

Käsikäyttäjänvalikkona on mukana oleva valinta.
 

### Asennus

```bash
pip install readme-rosetta
```

*Huomio: Edellyttää [Ollama](https://ollama.ai/) käytön asetusta ja toimintaa oma koneella.*

### Kansantarkoitukset

| Toiminto | Kuvaus | Alustavat |
| :--- | :--- | :--- |
| `path` | Lähde tiedosto tai projektiin liittyvä tehtäväkorteja | `README.md` |
| `--langs` | Lisätietojen listaus (esim. `es fr de`). | `[]` |
| `--src-lang` | Lähde kieli koodi. | `en` |
| `--model` | Ollama mallin ID käytetään. | `llama3.2` |
| `--readme` | Päättötehtävä tiedosto, jossa kääntettävät kieli lopettaavat. | `README.md` |
| `--no-split` | Lisää käännökset yhteen tietostoon. | `False` |
| `--dry-run` | Simulaattoritapaus, jossa ei kirjoita siihen kääntäviä tiedostoja. | `False` |
| `--verbose` | Matalapaineellinen tallennus tarkastelulaitteelle. | `False` |

---

## 📚 Sphinx integraatio

Toimittaa projectin tietokantaan professionaalia asiakaspyynnässä mukana olevia sähköistä käsitteitä.

Riippuen valinnasta:
1. **Keskitiedoston asentaminen:** Luovuttaa `docs/`-tiedostoa, jos sitä ei ole vielä olemassa.
2. **I18n-asetus:** Updataati tietoaineiston `conf.py` -sivulle `locale_dirs` ja `gettext` asetukset.
3. **Purkaisi sähköiset käsitteet:** Kirjaamassa kaikki `gettext`-sähköisiä käsitteitä projectin documentaatioon.
4. **PO-tiedostojen kääntäminen:** Käyttää LLM:tä järjestämään `.po`-tiedostot, jotka ovat asetettu `:role:` tai `.. directive::` mukaisesti.
5. **HTML:hen konverteerata:** Vastaa automatistamalla käännöksiä `SUMMARY.md` tiedostoon jossa oletetaan `es`-tiedoston sijainti, joka koostuu listojen nimet ja valinnat `Spanish`.
 
```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---

## 📖 GitBook tukeva

Yhdistää eri kieliä kaunean tiivisteen kehittämiseen.

Tietystä `--gitbook`-koodista:
- **Sähköinen yhteydenotoitus:** Linkitse kohdalla Introduction oikeaan `SUMMARY.md`-tiedostoon ja luo siihen liittyviä listat kaikilla käännöksillä.
- **Kieli-nimiin käyttäytyminen:** suhtautuu kieli-koodiin `es` mukaan itselleen.

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---

## ⚙️ Sijaintikokoelma

Luovuttaa projectiin muutaman arvon, jotta käyttäjät voivat tehdä vähän työrautena kääntötyön asennukseen.

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```
