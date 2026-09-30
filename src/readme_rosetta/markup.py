"""
Detecting markup the model made up.

All real inline markup (HTML tags, code spans, link targets, images) is
replaced by ⟦n⟧ tokens before a segment reaches the model, so any such
markup in the reply that the protected source did not contain was invented.
Harmless wrappers around the whole reply (``<span …>…</span>``, a code fence,
quotes) are unwrapped; anything else is rejected so the segment is retried.
"""

import re
from collections import Counter
from typing import List

_TAG = re.compile(r"</?([A-Za-z][A-Za-z0-9-]*)(?:\s[^<>]*)?/?>")
_WRAPPERS = [
    re.compile(r"^\s*<([A-Za-z][A-Za-z0-9-]*)(?:\s[^<>]*)?>(.*)</\1>\s*$", re.DOTALL),
    re.compile(r"^\s*```[\w-]*\n(.*?)\n```\s*$", re.DOTALL),
    re.compile(r'^\s*"(.*)"\s*$', re.DOTALL),
    re.compile(r"^\s*“(.*)”\s*$", re.DOTALL),
    re.compile(r"^\s*«\s?(.*?)\s?»\s*$", re.DOTALL),
]

# (description, pattern) for Markdown constructs that only appear if invented.
_CONSTRUCTS = [
    ("a code span", re.compile(r"`[^`\n]+`")),
    ("a link", re.compile(r"\]\([^)\s]+\)")),
    ("an image", re.compile(r"!\[")),
    ("a heading", re.compile(r"^\s{0,3}#{1,6}\s", re.MULTILINE)),
    ("a table row", re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)),
]


def unwrap(protected: str, out: str) -> str:
    """Strips wrappers around the whole reply that the source did not have."""
    for _ in range(3):
        for wrapper in _WRAPPERS:
            m = wrapper.match(out)
            if m and not wrapper.match(protected):
                out = m.group(m.lastindex).strip()
                break
        else:
            return out
    return out


def invented_markup(protected: str, out: str) -> str:
    """Describes markup in ``out`` that ``protected`` does not have, or ''."""
    extra_tags = Counter(t.lower() for t in _TAG.findall(out)) - Counter(
        t.lower() for t in _TAG.findall(protected)
    )
    problems: List[str] = []
    if extra_tags:
        tags = ", ".join(f"<{t}>" for t in sorted(extra_tags))
        problems.append(f"HTML tags that are not in the source ({tags})")
    for name, pattern in _CONSTRUCTS:
        if len(pattern.findall(out)) > len(pattern.findall(protected)):
            problems.append(name)
    return "adds " + ", ".join(problems) if problems else ""
