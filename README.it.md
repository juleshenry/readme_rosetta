#  README Rosetta

**README Rosetta** è un potente strumento di automazione progettato per tradurre la tua documentazione in diverse lingue utilizzando locali LLMs via [Ollama](https://ollama.ai/). Assicura che il tuo progetto sia accessibile ad un pubblico globale senza sacrificare la perfetta conformità ai formati Markdown e alla struttura del documento.

---

##  README Traduzione

README Rosetta si specializza nel rendere tuo progetto GitHub internazionale con minimali sforzi.

- **Supporto multilingue:** Traduci `README.md` in decine di lingue contemporaneamente.
- **Tabella navigazione:** Aggiungi automaticamente una "pietra" (tabella) di navigazione alla parte superiore del tuo README, consentendo ai utenti di passare velocemente tra le lingue.
- **Modalità flessibili:**
    - **Modalità a file separati (Default):** Genera files separati (ad esempio `README.es.md`, `README.fr.md`) per una struttura del progetto pulita.
    - **Modalità unitaria (`--no-split`):** Congiunge tutte le traduzioni in un file principale `README.md`, separato da commenti HTML.
- **Preservazione dei formati Markdown:** Gestisce intelligentemente i titoli, elenchi e blocchi di codice per garantire che l'output tradotto rimanga funzionale e ben formato.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---

##  Interfaccia di linea di comando (CLI)

L'interfaccia è progettata per essere intuitiva ma potente.

### Installazione

```bash
pip install readme-rosetta
```

*Nota: Richiede [Ollama](https://ollama.ai/) da installare e eseguire sulla tua macchina.*

### Opzioni globali

| Opzione | Descrizione | Valore predefinito |
| :--- | :--- | :--- |
| `path` | Percorso del file di origine o della cartella di progetto. | `README.md` |
| `--langs` | Elenco dei codici delle lingue target (ad esempio `es fr de`). | `[]` |
| `--src-lang` | Codice della lingua fonte. | `en` |
| `--model` | ID del modello Ollama da utilizzare. | `llama3.2` |
| `--readme` | Percorso del file di output principale README. | `README.md` |
| `--no-split` | Aggiungi le traduzioni in un solo file. | `False` |
| `--dry-run` | Simula il processo senza scrivere file. | `False` |
| `--verbose` | Abilita la loggistica dettagliata per l' debug. | `False` |

---

##  Integrazione con Sphinx

Aumenta le tue documentazioni a livelli professionali con il supporto automatico di i18n di Sphinx.

Quando esegui con la bandiera `--sphinx`, README Rosetta:
1.  **Inizializza Sphinx:** Crea una directory `docs/` se non esiste.
2.  **Auto-configura l'i18n:** Aggiorna `conf.py` con le impostazioni necessarie `locale_dirs` e `gettext`.
3.  **Estrae le stringhe:** Esegui `gettext` per trovare tutte le stringhe traducibili nel tuo documento.
4.  **Traduci i file PO:** Usa il modello LLM per tradurre `.po`, preservando lo specifico sintassi di Sphinx come `:role:` o `.. directive::`.
5.  **Crea HTML:** Genera automaticamente le costruzioni HTML localizzate per ogni lingua target.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---

##  Supporto GitBook

Mantieni a buon punto la tua documentazione multilingue con GitBook.

La bandiera `--gitbook` genera un file `SUMMARY.md` che mappa le tue README tradotte in una struttura compatibile con la navigazione di GitBook.

- **Auto-linking:** Collega l'Introduzione alla tua principale README e crea liste di item per ogni versione tradotta.
- **Nomina delle lingue:** Risolve automaticamente i codici delle lingue (come `es`) nelle loro nomi pieni (come `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---

##  Configurazione

Economizza tempo definendo i tuoi standard del progetto in `pyproject.toml`:

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```
