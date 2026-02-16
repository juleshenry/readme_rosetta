# 🗿 README Rosetta

**README Rosetta** es un poderoso herramienta de automatización diseñada para traducir tus documentación en múltiples idiomas utilizando LLMs locales mediante [Ollama](https://ollama.ai/). Asegura que tu proyecto esté accesible a una audiencia global mientras se mantiene la perfecta formática Markdown y estructura de documento.
## 🌍 README Traducción

README Rosetta se especializa en hacer que su proyecto de GitHub internacional con mínimo esfuerzo.

- **Apoyo multilenguaje:** Traduce `README.md` a docenas de idiomas simultáneamente.
- **Tabla de navegación:** Agrega automáticamente una "piña" (tablas) en la parte superior de su README, permitiendo a los usuarios cambiar rápidamente entre idiomas.
- **Modos flexibles:
    - **Modo dividido (Predeterminado):** Genera archivos separados (por ejemplo, `README.es.md`, `README.fr.md`) para una estructura de proyecto limpio.
    - **Modo unificado (`--no-split`):** Agrega todas las traducciones al archivo principal `README.md`, separadas por comentarios HTML.
- **Preservación del formato Markdown:** Maneja inteligentemente los títulos, listas y bloques de código para asegurar que el salida traducida se mantenga funcional y bien formada.

```bash
# Translate README.md to Spanish, French, and German
readme-rosetta --langs es fr de
```

---
## CLI

La interfaz de línea de comandos es diseñada para ser intuitiva pero poderosa.
### Instalación

```bash
pip install readme-rosetta
```

*Nota: Requiere que se instale y ejecute [Ollama](https://ollama.ai/) en su sistema.*
### Opciones globales

| Opción | Descripción | Defecto |
| :--- | :--- | :--- |
| `path` | Ruta del archivo de fuente o directorio de proyecto. | `README.md` |
| `--langs` | Lista de códigos de idioma objetivo (por ejemplo, `es fr de`). | `[]` |
| `--src-lang` | Código del idioma de fuente. | `en` |
| `--model` | ID del modelo Ollama a utilizar. | `llama3.2` |
| `--readme` | Ruta al archivo principal README con las traducciones. | `README.md` |
| `--no-split` | Añadir traducciones a un solo archivo. | `False` |
| `--dry-run` | Simular el proceso sin escribir archivos. | `False` |
| `--verbose` | Habilitar registros detallados para depuración. | `False` |
## 📚 Integración de Sphinx

Aumenta la escalabilidad de tus documentaciones a niveles profesionales con el soporte automático de i18n de Sphinx.

Cuando ejecutas con la bandera `--sphinx`, README Rosetta:
1.  **Inicializa Sphinx:** Establece una carpeta `docs/` si no existe.
2.  **Autoconfirma i18n:** Actualiza `conf.py` con las configuraciones necesarias de `locale_dirs` y `gettext`.
3.  **Extrae cadenas:** Corre `gettext` para encontrar todas las cadenas traducibles en tus documentaciones.
4.  **Traduce archivos PO:** Utiliza el LLM para traducir los archivos `.po`, preservando la sintaxis específica de Sphinx como `:role:` o `.. directive::`.
5.  **Edifica HTML:** Genera automáticamente las ediciones en HTML locales para cada idioma objetivo.

```bash
# Setup Sphinx with translations for Spanish and Japanese
readme-rosetta --sphinx --langs es ja
```

---
## 📖 Soporte de GitBook

Mantener fácilmente un libro de GitBook en varias lenguas.

La bandera `--gitbook` genera un archivo `SUMMARY.md` que mapea tus README traducidos a una estructura compatible con la navegación del libro de GitBook.

- **Enlaces automáticos:** Enlaza la Introducción a tu principal README y crea ítems de lista para cada versión traducida.
- **Nombres de lenguajes:** Resuelve automáticamente los códigos de idioma (como `es`) en sus nombres completos (como `Spanish`).

```bash
# Generate localized READMEs and a SUMMARY.md for GitBook
readme-rosetta --gitbook --langs hi zh pt
```

---
## ⚙️ Configuración

Guarda tiempo definiendo tus opciones de proyecto en `pyproject.toml`:

```toml
[tool.readme-rosetta]
model = "llama3.2"
src-lang = "en"
langs = ["es", "fr", "de"]
path = "README.md"
sphinx = true
gitbook = false
```
⚠️ Tratamiento de errores y limitaciones

El traducción automática utilizando LLMs es poderosa pero puede introducir artefactos de formato ocasionalmente, especialmente en entornos Sphinx/RST complejos.
### Problemas Comunes
- **Desalineación de Citaciones:** Los LLMs pueden fallar al cerrar una cadena `` `` `` or `` `` `` sin citación.
- **Longitudes de Encabezados:** Si un LLM agrega titulación en negrita (`**`) a un título, la línea subrayada de Sphinx puede no coincidir con la longitud del texto.
- **Sobrerepresentaciones Estruturales:** El modelo puede intentar agregar sus propias resúmenes o bloques de código "ayudosos" que no están en la fuente.
### Script de limpieza
Proporcionamos un script utilidad para identificar y eliminar errores comunes de traducción en tus archivos `.po`. Si se elimina una traducción, Sphinx simplemente se volverá a los textos originales del inglés para esa cadena.

```bash
# Run the cleanup utility
python3 scripts/cleanup_translations.py
```

*Nota: Siempre revisa tus construcciones de documentación. Aunque Rosetta busca perfección, la corrección manual de los archivos locales `.po` es a veces necesaria para documentación crítica.*
## 📜 Licencia

Este proyecto está licenciado bajo la Licencia MIT - vea el archivo [LICENSE](LICENSE) para detalles.
