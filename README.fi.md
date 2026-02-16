# 🗿 README Rosetta

**README Rosetta** on voimakas automaatioväline, jota käytetään suurten dokumenttien kääntämiseksi moniin kielimiin käyttämällä paikallisia LLMs [Ollama](https://ollama.ai/). Se varmistaa, että asiakasohjelma on saatavilla maailmanlaajuiselle yleisölle, kun taas säilytettään täydellinen Markdown- muotoilu ja dokumentin rakennetta.

---
## 🌍 README Kääntäminen

README Rosetta eristää GitHub-projektiasi kansainvälisiksi vähän työllystä.

- **Yhteiskielet:** Käännökäsännet `README.md` satoa yli kymmeniin kielisiin samanaikaisesti.
- **Navigointitaulu:** Automaattisesti etenee "pylväinen" (taulu) esikoitsijan arvoon tiedostasi, mahdollistamalla nopean käännöksen välitys.
- **Monimutkaiset Miehet:
    - **Pitelevä Mies (Kohtalaus):** Luo erilliset tiedostot (esim. `README.es.md`, `README.fr.md`) ehdollista suunnittelua varten.
    - **Yksittäinen Mies (`--no-split`):** Lisät käännökset päätiivistössä (`README.md`), erotettuna HTML-kommenteilla.
- **Merkintätaulu Sääntely:** Käsittelää headeriä, listaa ja koodi blokeja tarkkaan, varmistuen sen tulosta on toimivainen ja hyvin muotoilta.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---
## 🚫 Puhdistussuunnitelma (CLI)
Komentoalue on suunniteltu yksinkertaisuuden ja voimakkuuden vuoksi.
### Asentaminen

```bash
pip install readme-rosetta
```

*Merkin: Requiereinstallaation [Ollama](https://ollama.ai/) järjestelmään.*
 
  (Note: Original text could not be translated)
### Global Options

| Oletusarvo | Kuvaus | Väliarvo |
| :--- | :--- | :--- |
| `path` | Keskitallelu tai projektiin liittyvä lähdekilta. | `README.md` |
| `--langs` | Sana- listaus koodien (esim. `es fr de`). | `[]` |
| `--src-lang` | Lähtökieli-koodi. | `en` |
| `--model` | Ollama -mallin ID, jota käytetään. | `llama3.2` |
| `--readme` | Päätyön maine-esimerkin sijainti. | `README.md` |
| `--no-split` | Lopetetaan käännökset yhteen tekstejä. | `False` |
| `--dry-run` | Simuloitava prosessi ilman kirjoittamista tiedostoon. | `False` |
| `--verbose` | Aktivoituva laajennettu rekisteröinti toistetulle käytön varjolle. | `False` |
## 📚 Sphinx Integraatio

Keskittäkö dokumentoistasi ammattimaisia tasoja automaattisesti sisäiselläi18n tukeella Sphinxilla.

Jos käytät `--sphinx`-laitetta, README Rosetta:
1.  **Alkaa Sphinx:** Vastaavit ROSETTA CB_-rajauksen oikeaan osoitteeseen.
2.  **Autokonfiguroi i18n:** Päivittää ROSETTA CB_-tiedostot ROSETTA CB_- ja ROSETTA CB_-asemilla välttämättömiin asetuksiin.
3.  **Palaute merkkijonoista:** Kutsuu ROSETTA CB_ -ohjelmaa löytääkseen kaikki dokumentoissanne merkittyvät merkkijonot.
4.  **Käännökset PO-tiedostoihin:** Hyödyntää LLM-käyttöä kääntää ROSETTA CB_ -tiedostot, joten pitää tietysti tosiasia sphinxiin erikoinen syntaxi kuten ROSETTA CB- tai ROSETTA CB.
5.  **Kutsuu HTML-tallenteita:** Automaattisesti luovuttia sähköistä dokumentoista kaikkien lähtölanguan tavoitteisiin.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```
## 📖 GitBook Support

Easily maintain a multi-language GitBook.

Tämä rosettavalintu generoi `SUMMARY.md` tiedoston, joka määrittää, miten lukemassasi `--gitbook` tiedostoon kiinteää tietoa luettavista README-kirjoituksista GitBookin sivunnosten kanssa.

- **Automatiivinen Linkkejä:** Liittää esikolla kirjaamalla tärkeimmän README-tiedoston ja luodakseen listaesiemoita kaikkien käännöksiensä kohti.
- **Kielelliset nimet:** Kertoaa automaattisesti kielen koodi (kuten `es`) niiden täysimäisiksi nimiin (kuten `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---
## ⚙️ Konfiguraatio

Täytä aikaa asetuksien mukaisiin soveltumiseen `pyproject.toml` mukaan:

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```
⚠️ Harjoittelu ja rajoitukset

Automaattinen käännös LLM:illa on voimaa, mutta voi joutua tarpeen mukaan muuntamaan muovia, erityisesti monimutkaisissa Sphinx/RST-ympäristöissä.
### Yleiset Ongelmat
- **Väärässä Kirjoitettu Viivastus:** LLMs voi epäillä suljeutumista `` ` `` or ` `` ` ` -tyyppiseen merkkailuun.
- **Päävälisvuodoksinen Pituus:** Jos LLM tarjoaa korostettua (`**`) nimekseen, Sphinx allekirjoitus ei enää ole täsmällinen tekstin pituuteen.
- **Merkkailunmuotoiset Kansatukset:** Malleja voi yhdistää omia kertomuksiaan tai "auttavia" koodi blokeja, joita ei lähteessä ole.
### Puhdistus Skriitti
Tarjolla on käyttöohjeessa oleva laitteen avulla etsiä ja puhdistaa yleisimpiä käännösvirheitä `.po` tiedoissa. Jos käännös poistetaan, Sphinx selvittelee vain alkuperäisen englannin tekstin sijaisessa.

```bash
# Run the cleanup utility
python3 scripts/cleanup_translations.py
```

* huomio: Aina reviewaa dokumenttihenkilöitä. Rosetta pyrkii perfektioon, mutta käsikartuttaminen asetettuun `.po` -tiedoston paikkakunnalle on joskus tarpeen korkea-arvoisessa dokumentaatiossa.*
## 📜 Lisenssi

Tämä projekti on lisenssioodotuksena MIT-licensillä - katso [LICENSE](LICENSE) tiedosto säätelee lisenssiossa.
