"""
Segment-level translation cache.

Every translated segment is stored under a hash of everything that affects its
output (text, languages, backend, model, glossary, prompt version). Re-running
after editing a README therefore only sends the changed segments to the model,
and unchanged segments come back byte-for-byte identical — which keeps diffs of
the translated files readable. Commit the cache file to share it with CI.
"""

import json
import logging
import os
import tempfile
import threading
from typing import Dict, Iterable, Optional, Set

logger = logging.getLogger(__name__)

FORMAT_VERSION = 2


class Cache:
    def __init__(self, path: Optional[str]) -> None:
        self.path = path
        self.entries: Dict[str, str] = {}
        self.used: Set[str] = set()
        self.dirty = False
        self._lock = threading.Lock()
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("version") == FORMAT_VERSION:
                    self.entries = data.get("entries", {})
                else:
                    logger.info("Ignoring cache written by an older readme-rosetta.")
            except (OSError, ValueError, AttributeError) as e:
                logger.warning(f"Could not read cache {path}: {e}")

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            value = self.entries.get(key)
            if value is not None:
                self.used.add(key)
            return value

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self.entries[key] = value
            self.used.add(key)
            self.dirty = True

    def prune(self, targets: Iterable[str]) -> int:
        """Drops entries for ``targets`` that were not used in this run."""
        prefixes = tuple(f"{t}:" for t in targets)
        with self._lock:
            stale = [
                k for k in self.entries if k.startswith(prefixes) and k not in self.used
            ]
            for k in stale:
                del self.entries[k]
            if stale:
                self.dirty = True
            return len(stale)

    def save(self) -> None:
        if not self.path or not self.dirty:
            return
        with self._lock:
            directory = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(directory, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=directory, prefix=".cache-", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "version": FORMAT_VERSION,
                        "entries": dict(sorted(self.entries.items())),
                    },
                    f,
                    ensure_ascii=False,
                    indent=1,
                )
                f.write("\n")
            os.replace(tmp, self.path)
            self.dirty = False
