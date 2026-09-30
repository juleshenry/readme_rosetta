"""
Cheap, dependency-free checks that a translation is actually in the target
language.

Small local models regularly answer in the wrong language, leave the source
untouched, or drift into a third language halfway through a paragraph. We
cannot identify every language from a few words, but writing systems are easy
to tell apart, and that catches the most visible failures (an "Arabic" README
with Thai and Vietnamese in it).
"""

import re
from typing import Dict, Iterable, List, Optional, Tuple

from .lang_codes import get_language

# (start, end, script) — enough coverage for the languages in lang_codes.
_RANGES: List[Tuple[int, int, str]] = [
    (0x0041, 0x024F, "Latn"),
    (0x1E00, 0x1EFF, "Latn"),
    (0x0370, 0x03FF, "Grek"),
    (0x1F00, 0x1FFF, "Grek"),
    (0x0400, 0x052F, "Cyrl"),
    (0x0530, 0x058F, "Armn"),
    (0x0590, 0x05FF, "Hebr"),
    (0x0600, 0x06FF, "Arab"),
    (0x0750, 0x077F, "Arab"),
    (0x08A0, 0x08FF, "Arab"),
    (0xFB50, 0xFDFF, "Arab"),
    (0xFE70, 0xFEFF, "Arab"),
    (0x0900, 0x097F, "Deva"),
    (0x0980, 0x09FF, "Beng"),
    (0x0A00, 0x0A7F, "Guru"),
    (0x0A80, 0x0AFF, "Gujr"),
    (0x0B00, 0x0B7F, "Orya"),
    (0x0B80, 0x0BFF, "Taml"),
    (0x0C00, 0x0C7F, "Telu"),
    (0x0C80, 0x0CFF, "Knda"),
    (0x0D00, 0x0D7F, "Mlym"),
    (0x0D80, 0x0DFF, "Sinh"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x0E80, 0x0EFF, "Laoo"),
    (0x1000, 0x109F, "Mymr"),
    (0x10A0, 0x10FF, "Geor"),
    (0x1100, 0x11FF, "Hang"),
    (0x1200, 0x139F, "Ethi"),
    (0x1780, 0x17FF, "Khmr"),
    (0x3040, 0x309F, "Hira"),
    (0x30A0, 0x30FF, "Kana"),
    (0x3130, 0x318F, "Hang"),
    (0x3400, 0x4DBF, "Hani"),
    (0x4E00, 0x9FFF, "Hani"),
    (0xAC00, 0xD7AF, "Hang"),
    (0xF900, 0xFAFF, "Hani"),
]

# Composite writing systems: which character scripts count for them.
_COMPOSITE: Dict[str, Tuple[str, ...]] = {
    "Jpan": ("Hira", "Kana", "Hani"),
    "Kore": ("Hang", "Hani"),
}

# Placeholders, URLs and identifiers are expected to stay in Latin script.
_STRIP = re.compile(
    r"⟦\d+⟧|https?://\S+|\S+@\S+|\b[\w.-]+\.(?:py|md|js|ts|toml|json|ya?ml)\b"
)


def char_script(ch: str) -> Optional[str]:
    if not ch.isalpha():
        return None
    cp = ord(ch)
    for start, end, script in _RANGES:
        if start <= cp <= end:
            return script
    return "Other"


def script_counts(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ch in _STRIP.sub(" ", text):
        script = char_script(ch)
        if script:
            counts[script] = counts.get(script, 0) + 1
    return counts


def _expand(scripts: Iterable[str]) -> set:
    out = set()
    for s in scripts:
        out.update(_COMPOSITE.get(s, (s,)))
    return out


def check_language(
    source: str, translated: str, target: str, source_lang: str = "en"
) -> Optional[str]:
    """
    Returns a human-readable reason if ``translated`` does not look like
    ``target``, or ``None`` if it passes.
    """
    lang = get_language(target)
    expected = _expand(lang.scripts)
    src_counts = script_counts(source)
    out_counts = script_counts(translated)
    total = sum(out_counts.values())
    if total == 0:
        return None

    # 1. Contamination: letters from a script that is neither expected nor in
    #    the source (e.g. Thai inside an Arabic translation of English text).
    foreign = {
        s: n
        for s, n in out_counts.items()
        if s not in expected and s not in src_counts and s != "Latn"
    }
    if foreign:
        names = ", ".join(sorted(foreign))
        return f"contains text in an unexpected script ({names})"

    # 2. Non-Latin targets must be mostly written in their own script.
    #    Product names and identifiers legitimately stay in Latin, so only
    #    require a clear majority once there is enough text to judge.
    if "Latn" not in expected:
        own = sum(n for s, n in out_counts.items() if s in expected)
        if total >= 12 and own / total < 0.4:
            return f"is not written in {lang.name} script ({own}/{total} letters)"

    # 3. Latin targets: an unchanged multi-word sentence was not translated.
    if lang.code.split("-")[0] != source_lang.split("-")[0]:
        words = re.findall(r"[^\W\d_]{2,}", _STRIP.sub(" ", source))
        if len(words) >= 5 and _normalize(source) == _normalize(translated):
            return "is identical to the source text"

    return None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()
