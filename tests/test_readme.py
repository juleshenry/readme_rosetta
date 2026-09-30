from readme_rosetta.readme import (
    NAV_START,
    build_nav,
    build_unified,
    discover_translations,
    strip_rosetta,
    with_nav,
)

LEGACY = """<!-- <Original README.md> -->
# [Documentation Support in Multiple Languages](https://github.com/x/y/blob/main)
| About | |
| ------ | ---- |
| English | [Link to Head of Docs](#🗿-readme-rosetta) |
| Spanish | [Link to Head of Docs](README.es.md#🗿-readme-rosetta) |

# 🗿 README Rosetta

Body.

<!-- <Rosetta Translations> -->

# Old translation
"""


def test_strips_legacy_table_and_unified_sections():
    assert strip_rosetta(LEGACY) == "# 🗿 README Rosetta\n\nBody.\n"


def test_strip_is_idempotent_with_new_nav():
    text = with_nav("# T\n\nBody.\n", build_nav("en", ["es"], "README.md", "en"))
    assert NAV_START in text
    assert strip_rosetta(text) == "# T\n\nBody.\n"


def test_nav_uses_native_names_and_marks_current():
    nav = build_nav("es", ["ja", "es", "pt-BR"], "README.md", "en")
    assert (
        "🌐 [English](README.md) · **Español** · [日本語](README.ja.md) · [Português (Brasil)](README.pt-BR.md)"
        in nav
    )


def test_nav_base_url():
    nav = build_nav(
        "en", ["es"], "README.md", "en", base_url="https://github.com/o/r/blob/main/"
    )
    assert "[Español](https://github.com/o/r/blob/main/README.es.md)" in nav


def test_discover_ignores_unknown_codes(tmp_path):
    for name in [
        "README.md",
        "README.es.md",
        "README.hihn.md",
        "README.pt-BR.md",
        "README.backup.md",
    ]:
        (tmp_path / name).write_text("x")
    assert discover_translations(str(tmp_path / "README.md")) == ["es", "pt-BR"]


def test_unified_links_point_at_translated_headings():
    out = build_unified(
        "# 🗿 Demo\n\nText.\n", {"es": "# 🗿 Demo\n\nTexto.\n"}, "README.md", "en"
    )
    assert "[Español](#-demo-1)" in out
    assert "<!-- rosetta:translation:es -->" in out
    assert strip_rosetta(out) == "# 🗿 Demo\n\nText.\n"
