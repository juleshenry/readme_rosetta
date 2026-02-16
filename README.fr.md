# README Rosetta

**README Rosetta** est une puissante outil d'automatisation conçu pour traduire vos documents en plusieurs langues à l'aide de locaux LLMs via [Ollama](https://ollama.ai/). Il vous permet de rendre votre projet accessible à un public mondial tout en conservant les formats Markdown et la structure des documents parfaits.

---

## README Translation

README Rosetta est spécialisé dans la mise en langues internationales de votre projet GitHub avec effort minimum.

- **Multi-langue:** Traduisez vos `README.md` en dozens de langues simultanément.
- **Tableau de navigation:** Prépare automatiquement un "pierre" (table) de navigation à la tête de votre README, permettant aux utilisateurs de passer rapidement entre les langues.
- **Mode flexible:**
    - **Mode split (par défaut):** Génère des fichiers séparés (ex `README.es.md`, `README.fr.md`) pour une structure propre du projet.
    - **Mode unifié (`--no-split`):** Ajoute toutes les traductions au fichier principal `README.md`, séparées par des commentaires HTML.
- **Sauvegarde de Markdown:** Gère intelligemment les en-têtes, les listes et les code blocks pour assurer que l'output traduit reste fonctionnel et bien formaté.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---

## CLI Command Line Interface (CLI)

Le CLI est conçu pour être intuitif mais puissant.

### Installation

```bash
pip install readme-rosetta
```

*Note: Demande à Ollama d'être installé et exécuté sur votre système.*

### Options globales

| Option | Décription | Défaut |
| :--- | :--- | :--- |
| `path` | Chemin du fichier source ou du projet. | `README.md` |
| `--langs` | Liste des codes de langue cibles (ex `es fr de`). | `[]` |
| `--src-lang` | Code de la langue source. | `en` |
| `--model` | Identifiant du modèle Ollama à utiliser. | `llama3.2` |
| `--readme` | Chemin du principal fichier README d'output. | `README.md` |
| `--no-split` | Ajoute les traductions dans un seul fichier. | `False` |
| `--dry-run` | Simule le processus sans écrire de fichiers. | `False` |
| `--verbose` | Active la logique détaillée pour la débogage. | `False` |

---

## Sphinx Integration

Échellez vos documents à des niveaux professionnels avec une automatisation de Sphinx i18n.

Lorsque vous exécutez avec le drapeau `--sphinx`, README Rosetta:
1.  **Initialise Sphinx:** Crée un `docs/` si ce n'est pas déjà fait.
2.  **Configure les paramètres i18n:** Mise à jour `conf.py` avec les paramètres `locale_dirs` et `gettext` nécessaires.
3.  **Extraire les chaînes de strings:** Exécute `gettext` pour trouver toutes les chaînes traduisibles dans vos documents.
4.  **Translater PO Files:** Utilise le LLM pour traduire `.po`, en préservant la syntaxe Sphinx spécifique comme `:role:` ou `.. directive::`.
5.  **Génère HTML:** Génère automatiquement les builds HTML localisés pour chaque langue cible.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---

## GitBook Support

Maintenez avec facilité un projet à plusieurs langues GitBook.

Le drapeau `--gitbook` génère un `SUMMARY.md` qui mappait vos README traduits en une structure compatible avec la navigation de GitBook.

- **Autolinking:** Lien le Introduction au principal README et crée des items de liste pour chaque version traduite.
- **Noms de langues:** Résolve automatiquement les codes de langue (comme `es`) dans leurs noms complets (comme `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---

## ⚙️ Configuration

Économisez du temps en définissant vos paramètres de projet dans `pyproject.toml` :

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

## 📜 License

Ce projet est licensé sous le license MIT - consultez le [LICENSE](LICENSE) file pour les détails.
