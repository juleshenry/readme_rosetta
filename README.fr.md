#  README Rosetta (français)

ROSETTA CB
=====================================

Une plateforme open-source pour le travail de traduction automatique et manuelle.

Lancement
---------

Il est possible d'installer Rosetta sur votre système local à l'aide de Docker :
```
docker run -p 5000:5000 rosettacb/main
```

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
``` 
Exemple de résumé pour une tâche de traduction
---------------------------------------------

Pour créer un nouveau résumé, utilisez la commande suivante :
```bash
rosetta cb update /chemin/vers/dans/le/fichier.txt -o /chemin/vers/pour/depuis
```

```bash
pip install readme-rosetta
``` 
Exemple de modèle pour une tâche de traduction
---------------------------------------------

Utilisez la commande suivante :
```bash
rosetta cb create-model /chemin/vers/dans/le/fichier.txt --source-language=source_language --target-language=target_language
```

ROSETTA_RST_0 
Exemple de modèle pour une tâche de traduction
---------------------------------------------

Utilisez la commande suivante :
```bash
rosetta cb create-model /chemin/vers/dans/le/fichier.txt --source-language=source_language --target-language=target_language -m model.json
```

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
``` 
Exemple d'utilisation de l'API Rosetta
----------------------------------------

Utilisez la commande suivante :
```bash
curl 'http://127.0.0.1:5000/v1/translate' \
     -H 'Content-Type: application/json' \
     -d '{"source": "example source text", "target": "example target language code"}'
```

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
``` 
Exemple de modèle pour une tâche de traduction
---------------------------------------------

Utilisez la commande suivante :
```bash
rosetta cb update /chemin/vers/dans/le/fichier.txt --model=model.json
```

**README Rosetta** est une puissante outil d'automatisation conçu pour traduire votre documentation dans plusieurs langues en utilisant des LLM locaux via [Ollama](https://ollama.ai/). Cela assure que votre projet soit accessible à un public mondiau tout en conservant la formulation de police Markdown parfaite et la structure du document.

Aucun texte n'a été fourni.

## 🌎 L'Auteur
Lien vers l'auteur :```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
``` 

### Problèmes connus 
```bash
pip install readme-rosetta
``` : ```bash
pip install readme-rosetta
```

README Rosetta se spécialise à faire de votre projet GitHub internationale avec minimale effort.

- **Soutien multilingue :** Traduire `README.md` en dizaines de langues simultanément.
- **Tableau de navigation :** Préparer automatiquement une "pierre de navigation" (table) au sommet de votre README, permettant aux utilisateurs de passer rapidement entre les langues.
- **Modes flexibles :**
    - **Mode Split (Par défaut) :** Générer des fichiers séparés (par exemple `README.es.md`, `README.fr.md`) pour une structure de projet propre.
    - **Mode Uni (`--no-split`) :** Coller toutes les traductions au fichier principal `README.md`, séparées par des commentaires HTML.
- **Préservation du format Markdown :** Gérer intelligemment les titres, les listes et les blocs de code pour assurer que le sortie traduite reste fonctionnelle et bien formattée.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

Aucun texte n'a été fourni.

## Interface de ligne de commande (CLI)

Le CLI est conçu pour être à la fois intuitif et puissant.

### Installation 

1. Préparation des informations
2. Installation du logiciel de traduction
3. Configuration de l'interface utilisateur

```bash
pip install readme-rosetta
```

*Avertissement : Exige [Ollama](https://ollama.ai/) à être installé et exécuté sur votre système.*

**Options globales **

### Rosetta Initialization
```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
``` "Rosetta Initiative" ```bash
pip install readme-rosetta
``` "Début de l'initiative Rosetta"
ROSETTA RST 0 "Ressources disponibles" ROSETTA RST 1 "Ressources disponibles"

### Source Language Settings
ROSETTA RST 2 "Langue source définie sur l'utilisateur" ROSETTA RST 3 "Définition de la langue source sur le utilisateur"
ROSETTA RST 4 "L'anglais est la langue source par défaut dans ce contexte"

### Destination Language Settings
ROSETTA RST 5 "Langue cible définie par l'utilisateur" ROSETTA RST 6 "Définition de la langue cible sur le utilisateur"
ROSETTA RST 7 "Le français est la langue cible par défaut dans ce contexte"

### Translations
`--no-split` "Traduire toutes les ressources" `README.md` "Traduction des ressources"
`path` "Traduire uniquement le texte" `README.md` "Traduction du texte"
ROSETTA RST 12 "Traduire l'interface utilisateur" ROSETTA RST 13 "Personnalisation de la traduction"
ROSETTA RST 14 "Sélectionner une source de données" ROSETTA RST 15 "Intégrer des données externes"

### Advanced Options
ROSETTA RST 16 "Utiliser les méta-données pour la traduction" ROSETTA RST 17 "Personnalisation du modèle de traduction"
`llama3.2` "Supprimer toutes les annotations" `--readme` "Ajouter des annotations personnalisées"

### Error Handling
ROSETTA RST 20 "Afficher l'erreur de traduction" ROSETTA RST 21 "Resserrer le texte non traduit"
ROSETTA RST 22 "Ignorer toutes les erreurs de traduction"

| Option | Description | Default |
| :--- | :--- | :--- |
| `path` | Chemin de source d'un fichier ou d'une directory de projet. | `README.md` |
| `--langs` | Liste de codes de langue cibles (par exemple, `es fr de`). | `[]` |
| `--src-lang` | Code de langue source. | `en` |
| `--model` | ID du modèle Ollama à utiliser. | `llama3.2` |
| `--readme` | Chemin vers le fichier README principal d'output. | `README.md` |
| `--no-split` | Ajouter les traductions dans un seul fichier. | `False` |
| `--dry-run` | Simuler le processus sans écrire de fichiers. | `False` |
| `--verbose` | Activer l'affichage détaillé pour des opérations de débogage. | `False` |

Aucun texte n'a été fourni.

Sphinx intégration

Automatiser la mise à l'échelle de vos documents professionnels avec le soutien automatisé de Sphinx pour l'internationalisation.

Commencez par configurer le fichier `conf.py` pour inclure les ressources suivantes :

```python
# Raccourci pour le code source
source_suffix = ['.rst', '.md']

# Configuration globale pour Sphinx
copyright = u'2023, [Votre Nom]'
author = u'[Votre Nom]'
release = '1.0'

# Traductions
templates_path = ['_templates']
ext_modules = [
    sphinx.ext.autodoc,
    sphinx.ext.viewcode,
    sphinx.ext.todo,
]
language = {
    'fr_FR': None  # French (France)
}
```

Créez ensuite un nouveau fichier `_translators.yml` avec les traductions disponibles :

```yml
# Traductions
 translators:
  fr_FR: 
    - [Votre Nom] <[Adresse E-mail de Votre Nom]>
```

Ajoutez à votre document source la balise `.. translate::` pour spécifier le langage et l'auteur des traductions :

```rst
.. translate:: fr_FR
   :firstname: [Votre Prénom]
   :lastname: [Votre Nom]

Ce texte doit être traduit.
```

Utilisez le module Sphinx-Doctest pour tester les traductions et générer les données de documentation :

```python
# tests/integration/ doctest
"""
Test des traductions.

.. autosummary::

   translation

.. autodoc:: 
   :members:
```

Installez les dépendances nécessaires avec pip :

```bash
pip install sphinx sphinx-autodoc sphinx-doctest PyInquirer
```

Créez ensuite un fichier `Makefile` pour générer le document :

```makefile
SphinxDoc = build/sphinxdoc

.PHONY: all clean html
all: $(SphinxDoc)

clean:
    - rm -rf build tests
    - rm -f SphinxDoc

html:
    makehtml --no-coverage --build-dir=. sphinxdoc
```

Démarrez le processus de génération avec :

```bash
make all
```

Cela devrait créer la documentation HTML dans le répertoire `build/sphinxdoc`.

Lorsque vous exécutez avec la drapeau `--sphinx` , README Rosetta:

1.  **Initialise Sphinx** : Crée un dossier `docs/` si celui-ci n'existe pas.
2.  **Auto configure i18n** : Met à jour `conf.py` avec les paramètres nécessaires `locale_dirs` et `gettext`.
3.  **Extraire des chaînes de strings** : Exécute `gettext` pour trouver toutes les chaînes translatables dans votre documentation.
4.  **Traduire des fichiers PO** : Utilise LLM pour traduire `.po`, en conservant la syntaxe Sphinx spécifique comme `:role:` ou `.. directive::`.
5.  **Construire HTML** : Génère automatiquement les builds HTML localisés pour chaque langue cible.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

Aucun texte n'a été fourni.

### Support GitBook

Pour plus d'informations sur la conversion de votre document en format Markdown, consultez le guide [Conseils pour convertir un fichier en Markdown](```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```). Vous pouvez également consulter notre documentation pour plus d'informations sur les fonctionnalités et les limites du support GitBook.

Facilement maintenir un livre numérique multilingue en GitBook.

Le drapeau `--gitbook` génère un fichier `SUMMARY.md` qui cartographie vos README traduits dans une structure compatible avec la navigation de GitBook.

-   **Liaison Automatique :** Relie l'introduction à votre principal fichier README et crée des items de liste pour chaque version traduite.
-   **Nom des Langues :** Résout automatiquement les codes de langue (comme `es`) dans leurs noms complets (comme `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

Aucun texte n'a été fourni.

## 🛠️ Configuration 

(RÉCAPITULATION DE LA CONFIGURATION DU Système) 

(RÉSUMÉ DES PARAMÈTRES ACTUELLEMENT Utilisés)

(RAPPEL DES ÉTAPES DE CONFIGURATION précédentes)

Enregistrez du temps en définissant vos paramètres de projet par défaut dans `pyproject.toml` :

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
