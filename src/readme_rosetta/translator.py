"""
Segment translation: batching, validation, retries and caching.

Handlers split documents into small segments (a paragraph, a heading, a table
cell). The translator protects inline syntax in each segment, sends batches of
segments to the backend, and accepts a translation only if it passes every
check. Rejected segments are retried one by one with the rejection reason fed
back to the model. Segments that still fail keep their source text and are
recorded in ``failures`` so the CLI can report them and exit non-zero.
"""

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .backends import Backend, BackendError
from .cache import Cache
from .lang_codes import get_language
from .langcheck import check_language
from .markup import invented_markup, unwrap
from .protect import Protector, restore, token_problem

logger = logging.getLogger(__name__)

# Bump when the prompt or validation changes enough to invalidate old output.
PROMPT_VERSION = 1

BATCH_MAX_SEGMENTS = 12
BATCH_MAX_CHARS = 2500

_SEG_RE = re.compile(r'<seg id="(\d+)">(.*?)</seg>', re.DOTALL)
_PREAMBLE_RE = re.compile(
    r"^\s*(?:here(?:'s| is) (?:the |your )?translat\w*|translat\w*)[^\n]*:\s*\n",
    re.IGNORECASE,
)
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)

ProgressCallback = Callable[[int], None]


@dataclass
class Failure:
    target: str
    source: str
    reason: str


class Translator:
    def __init__(
        self,
        backend: Backend,
        source_lang: str = "en",
        cache: Optional[Cache] = None,
        glossary: Optional[Dict[str, Dict[str, str]]] = None,
        keep_terms: Sequence[str] = (),
        max_retries: int = 2,
        context: str = "",
        pause: float = 0,
    ) -> None:
        self.backend = backend
        self.source_lang = source_lang
        self.cache = cache or Cache(None)
        self.glossary = glossary or {}
        self.keep_terms = list(keep_terms)
        self.max_retries = max_retries
        self.context = context
        self.pause = pause
        self.failures: List[Failure] = []
        self.calls = 0
        self._lock = threading.Lock()
        self._protectors: Dict[str, Protector] = {}

    # ------------------------------------------------------------------ keys

    def _protector(self, syntax: str) -> Protector:
        if syntax not in self._protectors:
            # Do-not-translate terms stay visible: as opaque tokens the model loses
            # the sentence's meaning and tends to drop them. They are checked instead.
            self._protectors[syntax] = Protector(syntax)
        return self._protectors[syntax]

    def cache_key(self, text: str, target: str, syntax: str) -> str:
        material = json.dumps(
            [
                PROMPT_VERSION,
                self.backend.id,
                self.source_lang,
                target,
                syntax,
                self.keep_terms,
                self._terms(target),
                self.context,
                text,
            ],
            ensure_ascii=False,
            sort_keys=True,
        )
        return f"{target}:{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"

    # ---------------------------------------------------------------- public

    def needs_translation(self, text: str, syntax: str = "markdown") -> bool:
        protected, _ = self._protector(syntax).protect(text)
        for term in self.keep_terms:
            protected = protected.replace(term, "")
        return bool(_LETTER_RE.search(re.sub(r"⟦\d+⟧", "", protected)))

    def pending(
        self, texts: Sequence[str], target: str, syntax: str = "markdown"
    ) -> int:
        """Number of segments that would be sent to the model (for --dry-run)."""
        return sum(
            1
            for t in dict.fromkeys(texts)
            if self.needs_translation(t, syntax)
            and self.cache.get(self.cache_key(t, target, syntax)) is None
        )

    def translate(self, text: str, target: str, syntax: str = "markdown") -> str:
        return self.translate_many([text], target, syntax)[0]

    def translate_many(
        self,
        texts: Sequence[str],
        target: str,
        syntax: str = "markdown",
        progress: Optional[ProgressCallback] = None,
    ) -> List[str]:
        results: Dict[str, str] = {}
        todo: List[Tuple[str, str, List[str]]] = []  # (source, protected, saved)
        protector = self._protector(syntax)

        for text in dict.fromkeys(texts):
            if not self.needs_translation(text, syntax):
                results[text] = text
                continue
            cached = self.cache.get(self.cache_key(text, target, syntax))
            if cached is not None and self._still_valid(text, cached, target):
                results[text] = cached
                continue
            protected, saved = protector.protect(text)
            todo.append((text, protected, saved))

        if progress:
            progress(len(texts) - len(todo))

        for batch in self._batches(todo):
            outputs = self._translate_batch(batch, target)
            for (text, _, saved), out in zip(batch, outputs):
                if out is None:
                    results[text] = text
                    continue
                final = restore(out, saved)
                self.cache.set(self.cache_key(text, target, syntax), final)
                results[text] = final
            # Checkpoint after every batch so an interrupted run loses almost nothing.
            self.cache.save()
            if progress:
                progress(len(batch))

        return [results[t] for t in texts]

    # -------------------------------------------------------------- batching

    @staticmethod
    def _batches(todo: List[Tuple[str, str, List[str]]]):
        batch: List[Tuple[str, str, List[str]]] = []
        size = 0
        for item in todo:
            if batch and (
                len(batch) >= BATCH_MAX_SEGMENTS
                or size + len(item[1]) > BATCH_MAX_CHARS
            ):
                yield batch
                batch, size = [], 0
            batch.append(item)
            size += len(item[1])
        if batch:
            yield batch

    def _translate_batch(
        self, batch: List[Tuple[str, str, List[str]]], target: str
    ) -> List[Optional[str]]:
        system = self.system_prompt(target)
        request = "\n".join(
            f'<seg id="{i}">{p}</seg>' for i, (_, p, _) in enumerate(batch)
        )
        outputs: List[Optional[str]] = [None] * len(batch)
        reasons: Dict[int, str] = {}

        try:
            parsed = self._parse(
                self._call(system, [{"role": "user", "content": request}])
            )
        except BackendError:
            raise
        except (
            Exception
        ) as e:  # network hiccups etc.: fall through to per-segment retries
            logger.warning(f"Batch request failed: {e}")
            parsed = {}

        for i, (source, protected, saved) in enumerate(batch):
            out = parsed.get(i)
            if out is not None:
                out = unwrap(protected, out)
            reason = (
                "missing from the response"
                if out is None
                else self.problem(protected, out, len(saved), target)
            )
            if reason:
                reasons[i] = reason
            else:
                outputs[i] = out

        # Retry rejected segments individually, telling the model what was wrong.
        for i, reason in reasons.items():
            source, protected, saved = batch[i]
            outputs[i] = self._retry_single(
                system, protected, len(saved), target, reason, source
            )
        return outputs

    def _retry_single(
        self,
        system: str,
        protected: str,
        n_tokens: int,
        target: str,
        reason: str,
        source: str,
    ) -> Optional[str]:
        messages = [
            {
                "role": "user",
                "content": (
                    f'<seg id="0">{protected}</seg>\n\n(A previous translation of this segment '
                    f"was rejected because it {reason}.)"
                ),
            }
        ]
        for attempt in range(self.max_retries):
            try:
                reply = self._call(system, messages)
            except BackendError:
                raise
            except Exception as e:
                reason = f"request failed: {e}"
                continue
            out = self._parse(reply).get(0)
            if out is None and "<seg" not in reply:
                # One segment was asked for; a bare reply is still usable.
                out = _PREAMBLE_RE.sub("", reply).strip()
            if out is not None:
                out = unwrap(protected, out)
            reason = (
                "response was not in <seg> format"
                if out is None
                else self.problem(protected, out, n_tokens, target)
            )
            if not reason:
                return out
            logger.debug(f"Retry {attempt + 1} for {target} rejected: {reason}")
            messages += [
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": (
                        f"That translation was rejected: it {reason}. Translate the segment "
                        f"again into {get_language(target).name}. Reply only with "
                        '<seg id="0">…</seg> and keep every ⟦n⟧ token exactly once.'
                    ),
                },
            ]
        with self._lock:
            self.failures.append(Failure(target, source, reason))
        logger.warning(f"[{target}] kept source text ({reason}): {source[:60]!r}")
        return None

    def _call(self, system: str, messages: List[Dict[str, str]]) -> str:
        with self._lock:
            self.calls += 1
            first = self.calls == 1
        if self.pause and not first:
            time.sleep(self.pause)
        return self.backend.complete(system, messages)

    @staticmethod
    def _parse(reply: str) -> Dict[int, str]:
        reply = _PREAMBLE_RE.sub("", reply)
        return {int(i): body.strip() for i, body in _SEG_RE.findall(reply)}

    # ------------------------------------------------------------ validation

    def _still_valid(self, source: str, cached: str, target: str) -> bool:
        """
        Re-checks a cached translation, so entries written before a check
        existed are translated again instead of being reused forever.
        """
        # Judge the language on prose only, as the original check did.
        protector = self._protector("markdown")
        problem = invented_markup(source, cached) or check_language(
            protector.protect(source)[0],
            protector.protect(cached)[0],
            target,
            self.source_lang,
        )
        if not problem:
            for term in self.keep_terms:
                pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
                if len(re.findall(pattern, cached)) < len(re.findall(pattern, source)):
                    problem = f'lost "{term}"'
        if problem:
            logger.info(f"[{target}] re-translating cached segment ({problem})")
        return not problem

    def problem(self, protected: str, out: str, n_tokens: int, target: str) -> str:
        """Returns why ``out`` is not an acceptable translation, or ''."""
        if not out.strip():
            return "is empty"
        if "<seg" in out or "</seg" in out:
            return "contains nested <seg> tags"
        tokens = token_problem(out, n_tokens)
        if tokens:
            return f"has broken placeholders ({tokens})"
        markup = invented_markup(protected, out)
        if markup:
            return markup
        for term in self.keep_terms:
            pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
            if len(re.findall(pattern, out)) < len(re.findall(pattern, protected)):
                return f'translated or dropped "{term}", which must stay as is'
        if "\n" not in protected.strip() and "\n" in out.strip():
            return "split a single line into several lines"
        ratio = len(out) / max(len(protected), 1)
        if len(protected) > 40 and not 0.25 <= ratio <= 4:
            return (
                f"has a suspicious length ({len(out)} vs {len(protected)} characters)"
            )
        lang_issue = check_language(protected, out, target, self.source_lang)
        if lang_issue:
            return lang_issue
        return ""

    # ---------------------------------------------------------------- prompt

    def _terms(self, target: str) -> Dict[str, str]:
        return (
            self.glossary.get(target) or self.glossary.get(target.split("-")[0]) or {}
        )

    def system_prompt(self, target: str) -> str:
        src = get_language(self.source_lang).name
        tgt = get_language(target).name
        lines = [
            f"You are a professional technical translator localizing a software project's "
            f"documentation from {src} into {tgt}.",
            "",
            'The user sends segments formatted as <seg id="N">…</seg>. Reply with every segment '
            "translated, using the same ids and the same format, and nothing else.",
            "",
            "Rules:",
            f"- Write natural, idiomatic {tgt}, the way a native technical writer would. "
            f"Use only {tgt}; never switch to another language.",
            "- Tokens such as ⟦0⟧ stand for code, links or markup. Copy each token exactly once, "
            f"unchanged. You may move a token if {tgt} word order requires it.",
            "- Keep Markdown markers (**, *, _, ~~, [ and ]) around the same words.",
            "- Keep product names, commands, file names and other identifiers as they are.",
            "- Do not add, drop, summarize or explain anything. A segment that is one line "
            "stays one line.",
        ]
        if self.keep_terms:
            lines.append(
                "- Never translate these terms: " + ", ".join(self.keep_terms) + "."
            )
        terms = self._terms(target)
        if terms:
            lines.append("- Use these translations for recurring terms:")
            lines += [f"    {k} → {v}" for k, v in sorted(terms.items())]
        if self.context:
            lines += ["", f"Context: {self.context}"]
        return "\n".join(lines)
