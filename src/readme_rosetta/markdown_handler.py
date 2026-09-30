"""
Structure-aware Markdown translation.

The document is parsed with markdown-it-py. Only prose-bearing leaf blocks —
headings, paragraphs, table cells and the text lines of HTML blocks — are sent
for translation. Everything else (fenced and indented code, front matter, link
reference definitions, blank lines, table separators, list markers, blockquote
markers) is copied from the source byte for byte, so the model has no chance to
break it.
"""

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from markdown_it import MarkdownIt

from .slug import heading_anchors
from .translator import ProgressCallback, Translator

Translate = Callable[[str], str]

_FIRST_PREFIX = re.compile(
    r"^((?:[ \t]*>[ \t]?)*[ \t]*(?:(?:[-*+]|\d{1,9}[.)])[ \t]+(?:\[[ xX]\][ \t]+)?)?)"
)
_CONT_PREFIX = re.compile(r"^((?:[ \t]*>[ \t]?)*[ \t]*)")
_ATX = re.compile(
    r"^((?:[ \t]*>[ \t]?)*[ \t]*#{1,6}(?:[ \t]+|$))(.*?)((?:[ \t]+#+)?[ \t]*)$"
)
_HARD_BREAK = re.compile(r"^(.*?)( {2,}|\\|<br\s*/?>)$")
_RAW_HTML = re.compile(
    r"^\s*<(?:pre|script|style|code|textarea|!--|\?|!)", re.IGNORECASE
)
_HTML_LINE = re.compile(r"^(\s*)(.*?)(\s*)$")
_CODE_SPLIT = re.compile(
    r"(^[ \t]*```.*?^[ \t]*```|`[^`\n]+`)", re.MULTILINE | re.DOTALL
)
_ANCHOR_LINK = re.compile(r"(\]\(#|href=[\"']#)([^)\"'\s]+)")


@dataclass
class Unit:
    start: int
    end: int
    kind: str  # "paragraph" | "heading" | "setext" | "row" | "html"


def split_front_matter(text: str) -> Tuple[str, str]:
    """Splits YAML/TOML front matter off the top of a document."""
    m = re.match(
        r"^(---|\+\+\+)[ \t]*\n.*?\n(?:\1|\.\.\.)[ \t]*(?:\n|$)", text, re.DOTALL
    )
    if not m:
        return "", text
    return text[: m.end()], text[m.end() :]


class Document:
    def __init__(self, text: str) -> None:
        self.trailing_newline = text.endswith("\n")
        self.lines = text.split("\n")
        if self.trailing_newline:
            self.lines.pop()
        self.units = self._find_units(text)

    @staticmethod
    def _find_units(text: str) -> List[Unit]:
        md = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
        units: List[Unit] = []
        for tok in md.parse(text):
            if not tok.map:
                continue
            start, end = tok.map
            if tok.type == "heading_open":
                units.append(
                    Unit(
                        start,
                        end,
                        "heading" if tok.markup.startswith("#") else "setext",
                    )
                )
            elif tok.type == "paragraph_open":
                units.append(Unit(start, end, "paragraph"))
            elif tok.type == "tr_open":
                units.append(Unit(start, end, "row"))
            elif tok.type == "html_block" and not _RAW_HTML.match(tok.content):
                units.append(Unit(start, end, "html"))
        units.sort(key=lambda u: u.start)
        return units

    # ---------------------------------------------------------------- render

    def segments(self) -> List[str]:
        """All segments that would be sent to the translator, in document order."""
        found: List[str] = []

        def collect(s: str) -> str:
            found.append(s)
            return s

        self.render(collect)
        return found

    def render(self, tr: Translate) -> str:
        out: List[str] = []
        i = 0
        for unit in self.units:
            if unit.start < i:  # defensive: never process a line twice
                continue
            out.extend(self.lines[i : unit.start])
            out.extend(self._render_unit(unit, tr))
            i = unit.end
        out.extend(self.lines[i:])
        return "\n".join(out) + ("\n" if self.trailing_newline else "")

    def _render_unit(self, unit: Unit, tr: Translate) -> List[str]:
        lines = self.lines[unit.start : unit.end]
        if unit.kind == "heading":
            return [_render_atx(lines[0], tr)] + lines[1:]
        if unit.kind == "setext":
            return _render_paragraph(lines[:-1], tr) + lines[-1:]
        if unit.kind == "row":
            return [_render_row(line, tr) for line in lines]
        if unit.kind == "html":
            return [_render_html_line(line, tr) for line in lines]
        return _render_paragraph(lines, tr)


def _tr_keep_space(text: str, tr: Translate) -> str:
    m = _HTML_LINE.match(text)
    lead, core, trail = m.group(1), m.group(2), m.group(3)
    return lead + tr(core) + trail if core else text


def _render_atx(line: str, tr: Translate) -> str:
    m = _ATX.match(line)
    if not m or not m.group(2).strip():
        return line
    return m.group(1) + tr(m.group(2).strip()) + m.group(3)


def _render_paragraph(lines: List[str], tr: Translate) -> List[str]:
    first_prefix = _FIRST_PREFIX.match(lines[0]).group(1)
    parts = [(first_prefix, lines[0][len(first_prefix) :])]
    for line in lines[1:]:
        prefix = _CONT_PREFIX.match(line).group(1)
        parts.append((prefix, line[len(prefix) :]))

    if any(_HARD_BREAK.match(content) for _, content in parts[:-1]):
        # Hard line breaks are meaningful: translate line by line.
        out = []
        for prefix, content in parts:
            m = _HARD_BREAK.match(content)
            core, brk = (m.group(1), m.group(2)) if m else (content, "")
            out.append(
                prefix + (_tr_keep_space(core, tr) if core.strip() else core) + brk
            )
        return out

    # Soft-wrapped paragraph: translate as one sentence group, emit one line.
    text = " ".join(content.strip() for _, content in parts if content.strip())
    return [first_prefix + tr(text)] if text else lines


def split_row(line: str) -> List[str]:
    """Splits a table row on unescaped pipes outside code spans."""
    cells, buf = [], []
    i, ticks = 0, 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line):
            buf.append(line[i : i + 2])
            i += 2
            continue
        if ch == "`":
            run = len(line[i:]) - len(line[i:].lstrip("`"))
            ticks = 0 if ticks == run else (run if ticks == 0 else ticks)
            buf.append("`" * run)
            i += run
            continue
        if ch == "|" and not ticks:
            cells.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    cells.append("".join(buf))
    return cells


def _render_row(line: str, tr: Translate) -> str:
    return "|".join(
        _tr_keep_space(cell, tr) if cell.strip() else cell for cell in split_row(line)
    )


def _render_html_line(line: str, tr: Translate) -> str:
    return _tr_keep_space(line, tr) if line.strip() else line


class MarkdownHandler:
    """Translates Markdown documents segment by segment."""

    def __init__(self, translator: Translator) -> None:
        self.translator = translator

    def segments(self, text: str) -> List[str]:
        _, body = split_front_matter(text)
        return Document(body).segments()

    def translate(
        self, text: str, target: str, progress: Optional[ProgressCallback] = None
    ) -> str:
        front, body = split_front_matter(text)
        doc = Document(body)
        segments = doc.segments()
        translated = self.translator.translate_many(
            segments, target, "markdown", progress
        )
        mapping = dict(zip(segments, translated))
        return front + remap_anchors(body, doc.render(lambda s: mapping.get(s, s)))


def remap_anchors(source: str, translated: str) -> str:
    """
    Points in-page links (``[Install](#-installation)``) at the translated
    headings. Headings correspond one to one because structure is preserved.
    """
    src = [a for _, a in heading_anchors(source)]
    out = [a for _, a in heading_anchors(translated)]
    if len(src) != len(out):
        return translated
    mapping = {s: t for s, t in zip(src, out) if s != t}
    if not mapping:
        return translated

    def fix(chunk: str) -> str:
        return _ANCHOR_LINK.sub(
            lambda m: m.group(1) + mapping.get(m.group(2), m.group(2)), chunk
        )

    # Leave code (fenced blocks and inline spans) alone.
    parts = _CODE_SPLIT.split(translated)
    return "".join(p if i % 2 else fix(p) for i, p in enumerate(parts))
