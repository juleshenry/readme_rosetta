import re
from typing import Callable, Dict, List, Optional

import pytest

from readme_rosetta.backends import Backend
from readme_rosetta.cache import Cache
from readme_rosetta.translator import Translator

SEG = re.compile(r'<seg id="(\d+)">(.*?)</seg>', re.DOTALL)
WORD = re.compile(r"[^\W\d_]+")


def fake_translate(text: str, target: str) -> str:
    """Deterministic 'translation' that changes every word but keeps tokens and markup."""
    if target.startswith("ja"):
        return WORD.sub(lambda m: "あ" * len(m.group(0)), text)
    if target.startswith("ar"):
        return WORD.sub(lambda m: "ب" * len(m.group(0)), text)
    return WORD.sub(lambda m: m.group(0)[::-1] + "x", text)


class FakeBackend(Backend):
    """Answers the <seg> protocol. ``hook`` can corrupt specific segments."""

    name = "fake"

    def __init__(
        self, target: str = "es", hook: Optional[Callable[[str, int], str]] = None
    ):
        super().__init__("test")
        self.target = target
        self.hook = hook
        self.requests: List[List[Dict[str, str]]] = []

    def complete(self, system, messages):
        self.requests.append(messages)
        segs = SEG.findall(messages[0]["content"])
        out = []
        for i, body in segs:
            translated = fake_translate(body, self.target)
            if self.hook:
                translated = self.hook(translated, len(self.requests))
            out.append(f'<seg id="{i}">{translated}</seg>')
        return "\n".join(out)


@pytest.fixture
def make_translator(tmp_path):
    def make(target="es", hook=None, **kwargs):
        backend = FakeBackend(target, hook)
        cache = Cache(str(tmp_path / "cache.json"))
        return Translator(backend, cache=cache, **kwargs), backend

    return make
