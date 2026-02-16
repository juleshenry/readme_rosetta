# 🗿 README Rosetta

**README Roseta** es una herramienta de automatización poderosa diseñada para traducir tu documentación en múltiples idiomas utilizando LLMs locales a través de [Ollama](https://ollama.ai/). Asegura que tu proyecto sea accesible a una audiencia global mientras mantiene la estructura y formato de Markdown perfectos.

---

## 🌍 README Translation

README Roseta se especializa en hacer que tu proyecto GitHub internacional con mínimos esfuerzos.

- **Multi-lenguaje:** Traduce `README.md` a docenas de idiomas simultáneamente.
- **Tabla de navegación:** Agrega automáticamente una tabla "piedra" (tabla) al inicio de tu README, permitiendo a los usuarios cambiar rápidamente entre idiomas.
- **Modos flexibles:**
    - **Modo dividido (Defecto):** Genera archivos separados (por ejemplo, `README.es.md`, `README.fr.md`) para una estructura de proyecto limpia.
    - **Modo unificado (`--no-split`):** Añade todas las traducciones al archivo principal `README.md`, separadas por comentarios HTML.
- **Preservación del Markdown:** Maneja inteligentemente encabezados, listas y bloques de código para asegurar que el output traducido siga siendo funcional y bien formado.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---

## 🛠 Command Line Interface (CLI)

La CLI es diseñada para ser intuitiva pero poderosa.

### Instalación

```bash
pip install readme-rosetta
```

*Nota: Requiere que [Ollama](https://ollama.ai/) esté instalado y ejecutándose en tu sistema.*

### Opciones globales

| Opción | Descripción | Defecto |
| :--- | :--- | :--- |
| `path` | Ruta del archivo fuente o directorio de proyecto. | `README.md` |
| `--langs` | Lista de códigos de idioma objetivo (por ejemplo, `es fr de`). | `[]` |
| `--src-lang` | Código de lenguaje de fuente. | `en` |
| `--model` | ID del modelo Ollama a utilizar. | `llama3.2` |
| `--readme` | Ruta del archivo principal de README. | `README.md` |
| `--no-split` | Añadir traducciones a un solo archivo. | `False` |
| `--dry-run` | Simular el proceso sin escribir archivos. | `False` |
| `--verbose` | Habilitar registro detallado para depuración. | `False` |

---

## 📚 Sphinx Integración

Aumenta tus documentaciones a niveles profesionales con soporte automático de i18n de Sphinx.

Cuando se ejecuta con el bandera `--sphinx`, README Roseta:
1.  **Inicializa Sphinx:** Establece un directorio `docs/` si no existe.
2.  **Autoconfigura i18n:** Actualiza `conf.py` con las configuraciones necesarias `locale_dirs` y `gettext`.
3.  **Extraer cadenas:** Ejecuta `gettext` para encontrar todas las cadenas translatable en tus documentaciones.
4.  **Traducir archivos PO:** Utiliza el LLM para traducir `.po` files, preservando la sintaxis de Sphinx como `:role:` o `.. directive::`.
5.  **Crear HTML:** Genera alocaciones locales HTML para cada idioma objetivo.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---

## 📖 Soporte GitBook

Mantén fácilmente un proyecto multilingüe con GitBook.

La bandera `--gitbook` genera un archivo `SUMMARY.md` que mapea tus READMEs traducidos a una estructura compatible con la navegación de GitBook.

- **Enlaces automáticos:** Enlaza el Introducción al principal README y crea puntos de lista para cada versión traducida.
- **Nombres de idioma:** Resuelve automáticamente los códigos de idioma (como `es`) en sus nombres completos (como `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---

## ⚙️ Configuración

Ahorra tiempo definiendo tus configuraciones de proyecto en `pyproject.toml`:

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

## 📜 Licencia

Este proyecto está licenciado bajo la Licencia MIT - vea el archivo [LICENSE](LICENSE) para detalles.
