"""
GitHub-compatible heading anchors (a port of ``github-slugger``).
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)|\[([^\]]*)\]\[[^\]]*\]")
_HTML = re.compile(r"<[^>]+>")
_CODE = re.compile(r"`+([^`]*)`+")
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_EMPHASIS = re.compile(r"(\*\*|__|\*|_|~~)(?=\S)(.+?)(?<=\S)\1")


def heading_text(line: str) -> str:
    """Returns the visible text of an ATX heading line (``## [Foo](x) `bar```)."""
    text = line.strip()
    text = re.sub(r"^#{1,6}\s*", "", text)
    text = re.sub(r"\s+#+\s*$", "", text)
    text = _IMAGE.sub("", text)
    text = _LINK.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _CODE.sub(r"\1", text)
    text = _HTML.sub("", text)
    for _ in range(3):
        text = _EMPHASIS.sub(r"\2", text)
    return text.strip()


def slugify(text: str) -> str:
    """Slugs heading text the way GitHub does (no de-duplication)."""
    kept = []
    for ch in text.lower():
        cat = unicodedata.category(ch)
        if cat[0] in "LMN" or ch in "_- ":
            kept.append(ch)
    return "".join(kept).replace(" ", "-")


class Slugger:
    """Stateful slugger that de-duplicates repeated headings (``foo``, ``foo-1``)."""

    def __init__(self) -> None:
        self.seen: Dict[str, int] = {}

    def slug(self, text: str) -> str:
        base = slugify(text)
        slug = base
        while slug in self.seen:
            self.seen[base] += 1
            slug = f"{base}-{self.seen[base]}"
        self.seen[slug] = 0
        return slug


def heading_anchors(text: str) -> List[Tuple[int, str]]:
    """(line number, GitHub anchor) for every ATX heading outside code fences."""
    slugger = Slugger()
    out = []
    fence: Optional[str] = None
    for i, line in enumerate(text.split("\n")):
        m = _FENCE.match(line)
        if m:
            if fence is None:
                fence = m.group(1)[0] * 3
            elif line.strip().startswith(fence):
                fence = None
            continue
        if fence is None and re.match(r"^[ \t]{0,3}#{1,6}[ \t]+\S", line):
            out.append((i, slugger.slug(heading_text(line))))
    return out
