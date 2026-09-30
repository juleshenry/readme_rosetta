import os
import re

from readme_rosetta.markdown_handler import Document, MarkdownHandler, split_row

EXAMPLE = os.path.join(os.path.dirname(__file__), "examples", "zvec_README.md")

DOC = """---
title: Demo
---
# 🗿 Demo Project

Some **bold** text with `inline code` and a [link](https://example.com/a_b).
It continues here.

- First item with `code`
- [ ] A task
  1. Nested number
> Quoted wisdom
> across lines

Line with break  
second line

| Option | Description | Default |
| :--- | :--- | :--- |
| `--langs` | Target `a|b` languages. | `[]` |

```python
# comment that must stay
print("hello")
```

    indented code stays

<p align="center">
  <b>Centered</b> text
</p>

<!-- a comment -->

[ref]: https://example.com
Setext Title
============

See [Install](#install), not `[x](#install)`.

## Install
"""


def translate(text, make_translator, target="es"):
    translator, backend = make_translator(target)
    return MarkdownHandler(translator).translate(text, target), backend


def test_structure_is_preserved(make_translator):
    out, _ = translate(DOC, make_translator)
    lines = out.split("\n")

    assert out.startswith("---\ntitle: Demo\n---\n")  # front matter untouched
    assert "# 🗿 omeDx tcejorPx" in out  # heading level + emoji kept
    assert "**dlobx**" in out and "`inline code`" in out
    assert "(https://example.com/a_b)" in out
    assert "- tsriFx metix htiwx `code`" in out
    assert "- [ ] Ax ksatx" in out
    assert "  1. detseNx rebmunx" in out
    assert "> detouQx modsiwx ssorcax senilx" in out
    assert "eniLx htiwx kaerbx  " in lines  # hard break kept, line by line
    assert "| :--- | :--- | :--- |" in out
    assert "| `--langs` | tegraTx `a|b` segaugnalx. | `[]` |" in out
    assert '# comment that must stay\nprint("hello")' in out
    assert "    indented code stays" in out
    assert "  <b>deretneCx</b> txetx" in out
    assert "<!-- a comment -->" in out
    assert "[ref]: https://example.com" in out
    assert "txeteSx eltiTx\n============" in out


def test_in_page_links_follow_translated_headings(make_translator):
    out, _ = translate(DOC, make_translator)
    assert "## llatsnIx" in out
    assert "(#llatsnix)" in out
    assert "`[x](#install)`" in out


def test_segments_are_exactly_the_prose():
    segs = Document(DOC.split("---\n", 2)[2]).segments()
    assert "Target `a|b` languages." in segs
    assert not any("print(" in s for s in segs)
    assert not any(s.startswith("- ") or s.startswith(">") for s in segs)


def test_split_row_respects_code_and_escapes():
    assert split_row(r"| a | `x|y` | b \| c |") == [
        "",
        " a ",
        " `x|y` ",
        r" b \| c ",
        "",
    ]


def test_real_world_readme_keeps_every_code_block_and_url(make_translator):
    with open(EXAMPLE, encoding="utf-8") as f:
        src = f.read()
    out, _ = translate(src, make_translator)

    fence = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
    assert fence.findall(out) == fence.findall(src)
    urls = re.compile(r"https?://[^\s)\"'>]+")
    assert sorted(urls.findall(out)) == sorted(urls.findall(src))
    assert len(out.splitlines()) <= len(src.splitlines())

    def pipes_per_row(text):
        return [ln.count("|") for ln in text.splitlines() if ln.startswith("|")]

    assert pipes_per_row(out) == pipes_per_row(src)


def test_non_latin_target(make_translator):
    out, _ = translate(
        "# Hello World\n\nPlain text here.\n", make_translator, target="ja"
    )
    assert out == "# あああああ あああああ\n\nあああああ ああああ ああああ.\n"
