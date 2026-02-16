# 🗿 README Rosetta

**README Rosetta** es un herramienta poderosa de automatización diseñada para traducir tus documentación en múltiples idiomas utilizando LLM locales a través de [Ollama](https://ollama.ai/). Asegura que tu proyecto esté accesible a una audiencia global mientras mantiene perfectamente la formato Markdown y la estructura del documento.

---

## 🌍 README Translation

README Rosetta especializa en hacer que tu proyecto GitHub internacional con pocos esfuerzos.

- **Soporte de múltiples idiomas:** Traduce tus `README.md` en dos decenas de idiomas simultáneamente.
- **Tabla de navegación:** Añade automáticamente una tabla "pierna" (tabla) al top de tu README, permitiendo a los usuarios cambiar rápidamente entre idiomas.
- **Modos flexibles:**
    - **Modo dividido (Predeterminado):** Genera archivos separados (por ejemplo, `README.es.md`, `README.fr.md`) para una estructura de proyecto limpio.
    - **Modo unificado (`--no-split`):** Añade todas las traducciones al archivo principal `README.md`, separadas por comentarios HTML.
- **Preservación de Markdown:** Maneja inteligentemente cabezaleras, listas y bloques de código para asegurar que el salida traducida permanece funcional y bien formulada.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---

## 🛠 Command Line Interface (CLI)

La CLI está diseñada para ser intuitiva pero poderosa.

### Instalación

```bash
pip install readme-rosetta
```

*Nota: Requiere [Ollama](https://ollama.ai/) instalado y corriendo en tu sistema.*

### Opciones globales

| Opción | Descripción | Predeterminado |
| :--- | :--- | :--- |
| `path` | Ruta del archivo fuente o directorio de proyecto. | `README.md` |
| `--langs` | Lista de códigos de idioma objetivo (por ejemplo, `es fr de`). | `[]` |
| `--src-lang` | Código de idioma de fuente. | `en` |
| `--model` | ID del modelo Ollama a utilizar. | `llama3.2` |
| `--readme` | Ruta del archivo principal de README saliente. | `README.md` |
| `--no-split` | Añadir traducciones a un solo archivo. | `False` |
| `--dry-run` | Simular el proceso sin escribir archivos. | `False` |
| `--verbose` | Habilitar registro detallado para depuración. | `False` |

---

## 📚 Sphinx Integration

Aumenta tus documentaciones a niveles profesionales con soporte automático de i18n de Sphinx.

Cuando ejecutas con la bandera `--sphinx`, README Rosetta:
1.  **Inicializa Sphinx:** Establece un directorio `docs/` si no existe.
2.  **Autoconfigura el i18n:** Actualiza `conf.py` con las configuraciones necesarias `locale_dirs` y `gettext`.
3.  **Extrae Cadenas:** Corre `gettext` para encontrar todas las cadenas traducibles en tus documentación.
4.  **Traduce archivos PO:** Utiliza el LLM para traducir `.po` archivos, preservando el lenguaje específico de Sphinx como `:role:` o `.. directive::`.
5.  **Construye HTML:** Genera automáticamente ediciones HTML locales para cada idioma objetivo.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---

## 📖 GitBook Support

Mantén con facilidad una versión en español de tu GitBook.

La bandera `--gitbook` genera un archivo `SUMMARY.md` que mapea tus versiones salientes de README para una estructura compatible con la navegación de GitBook.

- **Enlazamiento automático:** Conecta el Introducción a tu principal README y crea listas de elementos para cada versión traducida.
- **Nombres de idioma:** Resuelve automáticamente los códigos de idioma (como `es`) en sus nombres completos (como `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---

## ⚙️ Configuración

Ahorra tiempo definiendo tus defaults del proyecto:

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
