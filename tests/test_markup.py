from readme_rosetta.markup import invented_markup, unwrap

SRC = "**README Rosetta** translates ⟦0⟧ into many languages."


def test_whole_reply_span_wrapper_is_unwrapped():
    # Seen from qwen2.5:7b on an Arabic README.
    out = '<span class="Apple-converted-space">**README Rosetta** يترجم ⟦0⟧</span>'
    assert unwrap(SRC, out) == "**README Rosetta** يترجم ⟦0⟧"


def test_code_fence_and_quote_wrappers_are_unwrapped():
    assert unwrap(SRC, "```markdown\nHola ⟦0⟧\n```") == "Hola ⟦0⟧"
    assert unwrap(SRC, '"Hola ⟦0⟧"') == "Hola ⟦0⟧"
    assert unwrap(SRC, "<p>«Hola ⟦0⟧»</p>") == "Hola ⟦0⟧"


def test_wrappers_the_source_has_are_kept():
    assert unwrap('"Quoted"', '"Citado"') == '"Citado"'


def test_inline_invented_tags_are_rejected():
    reason = invented_markup(
        SRC, "**README Rosetta** <b>traduce</b> ⟦0⟧ a <br/> idiomas."
    )
    assert "HTML tags that are not in the source (<b>, <br>)" in reason


def test_invented_markdown_is_rejected():
    assert "a code span" in invented_markup(SRC, "Traduce `README` ⟦0⟧.")
    assert "a link" in invented_markup(SRC, "Traduce [esto](https://x.io) ⟦0⟧.")
    assert "a heading" in invented_markup("Intro text", "# Texto de introducción")


def test_clean_translation_passes():
    assert (
        invented_markup(SRC, "**README Rosetta** traduce ⟦0⟧ a muchos idiomas.") == ""
    )
    # Comparisons in prose are not tags.
    assert invented_markup("if a < b", "si a < b") == ""


def test_stray_bracket_breaking_a_link_is_rejected():
    # Seen from qwen2.5:7b in German: "im [LICENSE]-Datei](LICENSE)".
    src = "See [LICENSE⟦0⟧ for details."
    assert "adds a closing bracket" in invented_markup(src, "Siehe [LICENSE]-Datei⟦0⟧.")
    assert invented_markup(src, "Siehe die Datei [LICENSE⟦0⟧.") == ""


def test_unclosed_bold_is_rejected():
    assert "bold" in invented_markup("**Fast** tool", "**Schnelles Werkzeug")
