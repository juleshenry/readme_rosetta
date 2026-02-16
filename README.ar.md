# 

**README Rosetta** هو أداة automation poderosa diseñada para traducir tu documentación en idiomas múltiples utilizando LLMs locales mediante [Ollama](https://ollama.ai/). garantiza que tu proyecto sea accesible a un público global mientras mantiene la estructura y el formato Markdown perfectos.

---

## 

README Rosetta se especializa en hacer que tu proyecto de GitHub internacional con minimal esfuerzo.

- **Soporte para múltiples idiomas:** traduce `README.md` en docenas de idiomas simultáneamente.
- **Tabla de navegación:** agrega automáticamente una tabla de navegación "piedra" (tabla) en la parte superior de tu README, permitiendo a los usuarios cambiar rápidamente entre idiomas.
- **Modos flexibles:**
    - **Modo separado (Por defecto):** genera archivos separados (por ejemplo, `README.es.md`, `README.fr.md`) para una estructura limpia del proyecto.
    - **Modo unificado (`--no-split`):** combina todas las traducciones en el archivo principal `README.md`, separadas por comentarios HTML.
- **Preservación de Markdown:** maneja inteligentemente los encabezados, listas y bloques de código para asegurar que la salida traducida permanece funcional y bien formada.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---

## 

La interfaz de línea de comandos (CLI) está diseñada para ser intuitiva pero poderosa.

### Instalación

```bash
pip install readme-rosetta
```

*Nota: Requiere que [Ollama](https://ollama.ai/) esté instalado y ejecutándose en tu sistema.*

### Opciones globales

| Opción | Descripción | Valor por defecto |
| :--- | :--- | :--- |
| `path` | Ruta del archivo fuente o directorio de proyecto. | `README.md` |
| `--langs` | Lista de códigos de idioma objetivo (por ejemplo, `es fr de`). | `[]` |
| `--src-lang` | Código del idioma fuente. | `en` |
| `--model` | ID del modelo Ollama a utilizar. | `llama3.2` |
| `--readme` | Ruta de la archivo principal de README saliente. | `README.md` |
| `--no-split` | Añadir traducciones a un solo archivo. | `False` |
| `--dry-run` | Simular el proceso sin escribir archivos. | `False` |
| `--verbose` | Habilitar registro detallado para depuración. | `False` |

---

## 

Integra tu documentación a niveles profesionales con soporte i18n automatizado de Sphinx.

Cuando ejecutas con la bandera `--sphinx`, README Rosetta:
1.  **Inicializa Sphinx:** crea un directorio `docs/` si no existe.
2.  **Configura automáticamente i18n:** actualiza `conf.py` con las configuraciones necesarias `locale_dirs` y `gettext`.
3.  **Extrae cadenas de traducción:** ejecuta `gettext` para encontrar todas las cadenas translatables en tu documentación.
4.  **Traduce archivos PO:** utiliza el LLM para traducir `.po` files, preservando la sintaxis específica del Sphinx como `:role:` o `.. directive::`.
5.  **Genera compilaciones HTML:** genera a automatismo las compilaciones HTML locales para cada idioma objetivo.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---

## 

Mantén fácilmente tu GitBook con múltiples idiomas.

La bandera `--gitbook` genera un archivo `SUMMARY.md` que mapea tus traducciones de README a una estructura compatible con la navegación del GitBook.

- **Enlazamiento automático:** enlaza la Introducción con tu archivo principal de README y crea listas para cada versión traducida.
- **Nombres de idiomas:** resuelve automáticamente los códigos de idioma (como `es`) a sus nombres completos (como `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---

## 

Ahorra tiempo definiendo tus preferencias del proyecto en `pyproject.toml`:

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```
