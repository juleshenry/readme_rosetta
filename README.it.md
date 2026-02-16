# 🗿 README Rosetta

**README Rosetta** è un potente strumento di automazione progettato per tradurre le tue documentazioni in diverse lingue utilizzando LLM locali via [Ollama](https://ollama.ai/). Garantisce che il tuo progetto sia accessibile a un pubblico globale mentre mantiene la perfetta formattazione Markdown e la struttura dei documenti.

---
## 🌍 README Traduzione

README Rosetta si specializza nel rendere il tuo progetto GitHub internazionale con minimali sforzi.

- **Supporto multi-lungua:** traduce `README.md` in decine di lingue contemporaneamente.
- **Tabella di navigazione:** aggiunge automaticamente una "piattaforma" (tabella) sopra la tua README, consentendo agli utenti di cambiare rapidamente tra le lingue.
- **Modalità flessibili:**
    - **Modo a pezzi (Predefinito):** genera file separati (ad es. `README.es.md`, `README.fr.md`) per una struttura progetto pulita.
    - **Modo unitario (`--no-split`):** aggiunge tutte le traduzioni al file principale `README.md`, separate da commenti HTML.
- **Preservazione di Markdown:** gestisce intelligentemente i titoli, liste e blocchi di codice per garantire che l'output tradotto rimanga funzionale e ben formato.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---
## Interfaccia di linea di comando (CLI)

L'interfaccia di linea di comando è progettata per essere intuitiva e potente.
### Installazione

```bash
pip install readme-rosetta
```

*Nota: Richiede [Ollama](https://ollama.ai/) da essere installato e eseguito sul tuo sistema.*
### Opzioni Globali

| Opzione | Descrizione | Predefinito |
| :--- | :--- | :--- |
| `path` | Percorso del file di origine o della cartella di progetto. | `README.md` |
| `--langs` | Elenco dei codici di lingua di destinazione (ad esempio, `es fr de`). | `[]` |
| `--src-lang` | Codice della lingua di origine. | `en` |
| `--model` | ID del modello Ollama da utilizzare. | `llama3.2` |
| `--readme` | Percorso del file principale README dell'output. | `README.md` |
| `--no-split` | Aggiungere traduzioni a un unico file. | `False` |
| `--dry-run` | Simulare il processo senza scrivere file. | `False` |
| `--verbose` | Abilitare il logging dettagliato per il debugging. | `False` |
## 📚 Integrazione di Sphinx

Aumenta la tua documentazione a livelli professionali con supporto automatico per l'internationalizzazione di Sphinx.

Quando esegui con la bandiera `--sphinx`, README Rosetta:
1.  **Inizia Sphinx:** Stabilizza una directory `docs/` se non esiste.
2.  **Auto-configuration dell'internationalizzazione:** Aggiorna `conf.py` con le impostazioni necessarie per `locale_dirs` e `gettext`.
3.  **Estrae Stringhe:** Esegui `gettext` per trovare tutte le stringhe traslatabili nella tua documentazione.
4.  **Traduce i file PO:** Utilizza l'LLM per tradurre i file `.po`, preservando lo specifico dello Sphinx come `:role:` o `.. directive::`.
5.  **Costrui è HTML:** Genera automaticamente gli edizioni HTML localizzate per ogni lingua di target.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---
## 📖 Support per GitBook multilingue

Mantenere facilmente una GitBook multilingua.

La bandiera `--gitbook` genera un file `SUMMARY.md` che mappa i tuoi README tradotti nella struttura compatibile con la navigazione di GitBook.

- **Autoimmatinazione dei Collegamenti:** Collega l'Introduzione al tuo principale README e crea elementi di lista per ogni versione tradotta.
- **Nomini delle Lingue:** Risolve automaticamente i codici di lingua (come `es`) nei loro nomi completi (come `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---
## ⚙️ Configurazione

Salva tempo definendo i tuoi default di progetto nella `pyproject.toml`:

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
⚠️ Problemi di risoluzione e limitazioni

La traduzione automatica con LLM è potente ma può introdurre artefatti di formattazione occasionalmente, soprattutto nei sistemi Sphinx/RST complessi.
### Problemi comuni
- **Mancata chiusura di backtick:** gli LLM potrebbero avere difficoltà a chiudere una stringa `` ` `` or ` `` ` `.
- **Lunghezze dei titoli:** se un LLM aggiunge sottolineature (`**`) a un titolo, l'underline della Sphinx non corrisponderà più alla lunghezza del testo.
- **Hallucinazioni strutturali:** il modello potrebbe cercare di aggiungere i suoi stessi riassunti o "aiutanti" codici blochi che non sono nel sorgente.
### Script Pulizia
Forniamo uno script utile per identificare e pulire errori comuni di traduzione nei tuoi file `.po`. Se una traduzione viene cancellata, Sphinx semplicemente caduto indietro al testo originale inglese per quella stringa.

```bash
# Run the cleanup utility
python3 scripts/cleanup_translations.py
```

*Nota: Sempre controllare le tue costruzioni della documentazione.  Rosetta mira a perfezione, ma la correzione manuale dei file `.po` localizzati è a volte necessaria per documentazioni critiche.*
## 📜 Licenza

Questo progetto è licenziato sotto la licenza MIT - vedi il file [LICENSE](LICENSE) per ulteriori informazioni.
