"""
LLM backends. A backend turns a system prompt plus a message list into text.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

Messages = List[Dict[str, str]]

DEFAULT_MODELS = {
    "ollama": "qwen2.5:7b",
    "anthropic": "claude-opus-5-5",
}


class BackendError(RuntimeError):
    """Raised when a backend cannot produce a response at all."""


class Backend:
    name = "base"

    def __init__(self, model: str) -> None:
        self.model = model

    @property
    def id(self) -> str:
        """Identifies backend+model in cache keys, so switching models re-translates."""
        return f"{self.name}:{self.model}"

    def prepare(self) -> None:
        """Checks the backend is reachable (and, for Ollama, pulls the model)."""

    def complete(self, system: str, messages: Messages) -> str:
        raise NotImplementedError


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, model: str, host: Optional[str] = None) -> None:
        super().__init__(model)
        import ollama

        self._ollama = ollama
        self.client = ollama.Client(host=host) if host else ollama.Client()

    def prepare(self) -> None:
        try:
            self.client.show(self.model)
        except self._ollama.ResponseError:
            logger.info(f"Pulling Ollama model {self.model} (first run only)...")
            self.client.pull(self.model)
        except Exception as e:
            raise BackendError(
                f"Cannot reach Ollama ({e}). Is it installed and running? "
                "See https://ollama.com/download"
            ) from e

    def complete(self, system: str, messages: Messages) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            options={"temperature": 0},
        )
        return response["message"]["content"]


class AnthropicBackend(Backend):
    name = "anthropic"

    def __init__(self, model: str, effort: str = "medium") -> None:
        super().__init__(model)
        try:
            import anthropic
        except ImportError as e:
            raise BackendError(
                "The anthropic backend needs the SDK: pip install 'readme-rosetta[anthropic]'"
            ) from e
        self._anthropic = anthropic
        # Credentials come from ANTHROPIC_API_KEY or an `ant auth login` profile.
        self.client = anthropic.Anthropic()
        self.effort = effort

    def complete(self, system: str, messages: Messages) -> str:
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=messages,
                output_config={"effort": self.effort},
                # On a safety decline, let the API retry on a suitable fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except (
            self._anthropic.AuthenticationError,
            self._anthropic.PermissionDeniedError,
            self._anthropic.NotFoundError,
        ) as e:
            raise BackendError(f"Anthropic API rejected the request: {e}") from e
        if response.stop_reason == "refusal":
            raise BackendError("the model declined to translate this segment")
        return "".join(b.text for b in response.content if b.type == "text")


def create_backend(
    name: str,
    model: Optional[str] = None,
    ollama_host: Optional[str] = None,
    effort: str = "medium",
) -> Backend:
    model = model or DEFAULT_MODELS.get(name)
    if name == "ollama":
        return OllamaBackend(model, host=ollama_host)
    if name == "anthropic":
        return AnthropicBackend(model, effort=effort)
    raise BackendError(
        f"Unknown backend '{name}' (choose from: {', '.join(DEFAULT_MODELS)})"
    )
