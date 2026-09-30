"""
Placeholder protection for inline syntax the model must not touch.

Before a segment goes to the model, code spans, URLs, HTML tags and similar
constructs are swapped for numbered tokens (``⟦0⟧``, ``⟦1⟧`` …). The model
translates the prose around them and the tokens are swapped back afterwards.
A translation is only accepted if every token comes back exactly once.
"""

import re
from typing import Dict, List, Sequence, Tuple

TOKEN_RE = re.compile(r"⟦(\d+)⟧")

_EMOJI = (
    r"[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B50\u2B55]\uFE0F?"
    r"(?:\u200D[\U0001F000-\U0001FAFF\u2600-\u27BF]\uFE0F?)*"
)

MARKDOWN_PATTERNS: Sequence[str] = (
    r"<!--[\s\S]*?-->",  # HTML comments
    r"``[^`]+``|`[^`\n]+`",  # inline code
    r"!\[[^\]]*\]\([^)]*\)",  # images (alt text included)
    r"\]\((?:[^()\s]|\([^()\s]*\))+(?:\s+\"[^\"]*\")?\)",  # link destinations
    r"\]\[[^\]]*\]",  # reference link labels
    r"\[\^[^\]]+\]",  # footnote references
    r"<(?:https?|mailto|ftp):[^>\s]+>",  # autolinks
    r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>",  # inline HTML tags
    r"https?://[^\s<>()\[\]]+[^\s<>()\[\].,;:!?'\"]",  # bare URLs
    r"&(?:[a-zA-Z]+|#\d+|#x[0-9a-fA-F]+);",  # HTML entities
    r"(?<![\w:]):[a-z0-9_+-]+:(?![\w:])",  # emoji shortcodes
    _EMOJI,  # emoji (models like to drop them, which also breaks anchors)
)

RST_PATTERNS: Sequence[str] = (
    r":[\w:-]+:`[^`]+`",  # :role:`target`
    r"``[^`]+``",  # inline literals
    r"`[^`]+ <[^>]+>`__?",  # `text <url>`_
    r"`[^`]+`__?",  # `target`_
    r"`[^`]+`",  # `interpreted text`
    r"\|[\w -]+\|",  # |substitution|
    r"^\.\. [\w:-]+::",  # directive markers
    r"https?://[^\s<>`]+[^\s<>`.,;:!?'\")]",  # bare URLs
)

PLAIN_PATTERNS: Sequence[str] = (r"https?://\S+",)

SYNTAXES: Dict[str, Sequence[str]] = {
    "markdown": MARKDOWN_PATTERNS,
    "rst": RST_PATTERNS,
    "plain": PLAIN_PATTERNS,
}


class Protector:
    """Builds a single regex from a syntax's patterns plus do-not-translate terms."""

    def __init__(
        self, syntax: str = "markdown", keep_terms: Sequence[str] = ()
    ) -> None:
        patterns = list(SYNTAXES[syntax])
        # Longest terms first so "README Rosetta" wins over "Rosetta".
        for term in sorted({t for t in keep_terms if t.strip()}, key=len, reverse=True):
            patterns.append(r"(?<!\w)" + re.escape(term) + r"(?!\w)")
        self.regex = re.compile("|".join(f"(?:{p})" for p in patterns), re.MULTILINE)

    def protect(self, text: str) -> Tuple[str, List[str]]:
        saved: List[str] = []

        def swap(match: "re.Match[str]") -> str:
            saved.append(match.group(0))
            return f"⟦{len(saved) - 1}⟧"

        # Existing token-like text in the source would confuse validation.
        text = text.replace("⟦", "⟦⁠")
        return self.regex.sub(swap, text), saved


def restore(text: str, saved: List[str]) -> str:
    def swap(match: "re.Match[str]") -> str:
        index = int(match.group(1))
        return saved[index] if index < len(saved) else match.group(0)

    return TOKEN_RE.sub(swap, text).replace("⟦⁠", "⟦")


def token_problem(translated: str, count: int) -> str:
    """Returns a description of missing/duplicated/invented tokens, or ''."""
    found = [int(i) for i in TOKEN_RE.findall(translated)]
    missing = sorted(set(range(count)) - set(found))
    extra = sorted({i for i in found if i >= count})
    dupes = sorted({i for i in found if found.count(i) > 1})
    problems = []
    if missing:
        problems.append("missing " + ", ".join(f"⟦{i}⟧" for i in missing))
    if dupes:
        problems.append("repeated " + ", ".join(f"⟦{i}⟧" for i in dupes))
    if extra:
        problems.append("invented " + ", ".join(f"⟦{i}⟧" for i in extra))
    return "; ".join(problems)
