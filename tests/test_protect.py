from readme_rosetta.protect import Protector, restore, token_problem


def test_markdown_inline_constructs_are_protected():
    text = (
        'Run `pip install x`, see [the docs](https://x.io/docs "Docs") and '
        "![logo](logo.png) or <https://x.io> <b>now</b> &amp; :rocket: https://a.b/c."
    )
    protected, saved = Protector("markdown").protect(text)
    for secret in (
        "pip install",
        "https://x.io/docs",
        "logo.png",
        "<b>",
        "&amp;",
        ":rocket:",
        "https://a.b/c",
    ):
        assert secret not in protected
    assert "[the docs" in protected  # link text stays translatable
    assert protected.endswith(".")  # trailing punctuation is not part of the URL
    assert restore(protected, saved) == text


def test_keep_terms_prefer_longest_match():
    protected, saved = Protector("markdown", ["Rosetta", "README Rosetta"]).protect(
        "Use README Rosetta."
    )
    assert saved == ["README Rosetta"]


def test_rst_roles_are_protected():
    protected, saved = Protector("rst").protect(
        "See :func:`foo` and ``bar`` or `link <http://x>`_."
    )
    assert saved == [":func:`foo`", "``bar``", "`link <http://x>`_"]


def test_token_problems():
    assert token_problem("⟦0⟧ ⟦1⟧", 2) == ""
    assert "missing ⟦1⟧" in token_problem("⟦0⟧", 2)
    assert "repeated ⟦0⟧" in token_problem("⟦0⟧ ⟦0⟧ ⟦1⟧", 2)
    assert "invented ⟦5⟧" in token_problem("⟦0⟧ ⟦5⟧", 1)


def test_literal_tokens_in_source_survive():
    protected, saved = Protector("markdown").protect("Literal ⟦0⟧ here")
    assert saved == []
    assert restore(protected, saved) == "Literal ⟦0⟧ here"


def test_emoji_are_protected():
    protected, saved = Protector("markdown").protect("🗿 README ✨ and 👩‍💻 dev")
    assert saved == ["🗿", "✨", "👩‍💻"]
