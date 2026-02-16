#  README Rosetta

**README Rosetta** est une puissante outil d'automatisation conçu pour traduire vos documents en plusieurs langues en utilisant des LLM locaux via [Ollama](https://ollama.ai/). Il garantit que votre projet soit accessible à un public mondiau tout en conservant une mise en forme de Markdown parfaite et la structure documentaire.
## 🌍 README Translation

README Rosetta spécialise en rendant votre projet GitHub international avec un effort minimum.

- **Multi-langue Support:** Traduire `README.md` dans des dizaines de langues simultanément.
- **Tableau de navigation:** Ajouter automatiquement une pierre de navigation (table) au sommet de votre README, permettant aux utilisateurs de passer rapidement entre les langues.
- **Modes flexibles:**
    - **Mode Split (Par défaut):** Générer des fichiers séparés (par exemple `README.es.md`, `README.fr.md`) pour une structure de projet propre.
    - **Mode Unifié (`--no-split`):** Ajouter toutes les traductions dans le fichier `README.md` principal, séparées par des commentaires HTML.
- **Préservation du Markdown:** Gérer intelligemment les en-têtes, les listes et les code blocks pour assurer que l'output traduit reste fonctionnel et bien formatté.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---
## 🛠 Interfaccé de ligne de commande (_CLI )

L'interfaccé de ligne de commande est conçu pour être intuitif tout en étant puissant.
### Installation

```bash
pip install readme-rosetta
```

*Remarque : Révèle nécessité [Ollama](https://ollama.ai/) à être installé et en exécution sur votre système.*
### Options Globales

| Option | Description | Défaut |
| :--- | :--- | :--- |
| `path` | Chemin vers le fichier source ou le répertoire de projet. | `README.md` |
| `--langs` | Liste de codes de langue cible (par exemple, `es fr de`). | `[]` |
| `--src-lang` | Code de langue source. | `en` |
| `--model` | Identifiant modèle Ollama à utiliser. | `llama3.2` |
| `--readme` | Chemin vers le fichier principal README d'output. | `README.md` |
| `--no-split` | Ajouter les traductions dans un seul fichier. | `False` |
| `--dry-run` | Simuler le processus sans écrire de fichiers. | `False` |
| `--verbose` | Activer un journal détaillé pour la débogage. | `False` |
## 📚 Intégration avec Sphinx

Amplifiez votre documentation à des niveaux professionnels grâce à un soutien automatique pour l'internationalisation de Sphinx.

Lorsque vous exécutez avec la drapeau `--sphinx`, README Rosetta:
1.  **Initialise Sphinx:** Met en place une `docs/` dans le répertoire si celui-ci n'existe pas.
2.  **Auto-configure l'internationalisation:** Mise à jour la `conf.py` avec les paramètres nécessaires pour `locale_dirs` et `gettext`.
3.  **Extraire les chaînes de traduction:** Exécute `gettext` pour trouver toutes les chaînes de traduction dans votre documentation.
4.  **Traduire les fichiers PO:** Utilisez l'LLM pour traduire les fichiers `.po`, en conservant la syntaxe spécifique à Sphinx comme `:role:` ou `.. directive::`.
5.  **Construction de l'HTML:** Génère automatiquement des builds HTML locaux pour chaque langue cible.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---
## 📖 Support GitBook

Maintenir facilement un livre multi-langue GitBook.

La drapeau `--gitbook` génère un fichier `SUMMARY.md` qui mapppe les READMEs traduits dans une structure compatible avec la navigation du livre.

- **Liaison automatique :** Ligne la Introduction au main README et crée des éléments de liste pour chaque version traduite.
- **Nom de langues :** Résout automatiquement les codes de langue (comme `es`) dans leurs noms complets (comme `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---
## ⚙️ Configuration

Sauve temps en définissant vos préférences de projet dans `pyproject.toml` :

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```
⚠️ Les erreurs de traduction automatique par utilisateurs d'LLMs peuvent être puissantes mais introduisent parfois des artefacts de formatage, en particulier dans les environnements Sphinx/RST complexes.
### Problèmes courants
* **Malentendus de backticks :** Les LLMs peuvent échouer à fermer une `` `` `` or `` `` `` chaîne.
* **Longueurs de titres :** Si un modèle ajoute du boulonnage (`**`) à un titre, la mise en saillie Sphinx peut ne pas plus correspondre à la longueur du texte.
* **Hallucinations structurales :** Le modèle peut essayer d'ajouter ses propres résumés ou des blocs de code "utiles" qui ne sont pas dans la source.
### Script de nettoyage
Nous fournissent un script utilitaire pour identifier et nettoyer les erreurs de traduction courantes dans vos fichiers `.po`. Si une traduction est nettoyée, Sphinx retombera simplement sur le texte original en anglais pour cette chaîne.

```bash
# Run the cleanup utility
python3 scripts/cleanup_translations.py
```

*Note: Révisez toujours vos builds de documentation. Même si Rosetta s'efforce de l'exactitude parfaite, la correction manuelle des fichiers `.po` localisés est parfois nécessaire pour les documents à haut risque.*
## 📜 Licence

Ce projet est licence sous la MIT Licence - consultez le fichier [LICENSE](LICENSE) pour plus de détails.
