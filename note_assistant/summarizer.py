"""Summarization backend — Apple Foundation Models, MLX (on-device), or Ollama."""
from __future__ import annotations

import platform
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, AsyncIterator

from .config import SummarizationConfig
from note_assistant import logger


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseSummarizer(ABC):
    @abstractmethod
    async def summarize(self, transcript: str) -> AsyncGenerator[str, None]:
        """Yield streaming summary tokens for the given transcript."""
        return
        yield  # pragma: no cover — marks this as an async generator


# ---------------------------------------------------------------------------
# Apple Foundation Models backend — placeholder (Task 3 will replace this)
# ---------------------------------------------------------------------------

class AppleFoundationSummarizer(BaseSummarizer):
    """Placeholder — replaced in Task 3 with apple_fm_sdk implementation."""

    def __init__(
        self,
        config: SummarizationConfig,
        language_input: str = "English",
        language_output: str = "English",
    ) -> None:
        raise RuntimeError(
            "Apple Foundation Models backend not yet configured. "
            "Complete Task 3 of the migration plan."
        )

    async def summarize(self, transcript: str) -> AsyncGenerator[str, None]:
        raise RuntimeError("Not implemented")
        yield  # pragma: no cover


# ---------------------------------------------------------------------------
# MLX backend — on-device LLM via Apple Silicon GPU/Neural Engine (Python-native)
# ---------------------------------------------------------------------------

class MLXSummarizer(BaseSummarizer):
    """On-device summarization using MLX (Apple Silicon GPU). No Xcode needed."""

    DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    def __init__(self, config: SummarizationConfig, language_input: str = "English", language_output: str = "English"):
        self.config = config
        self.language_input = language_input
        self.language_output = language_output
        self._model = None
        self._tokenizer = None
        self._load()

    def _load(self) -> None:
        try:
            from mlx_lm import load
        except ImportError as e:
            raise RuntimeError(
                "mlx-lm not installed. Run: uv add mlx mlx-lm"
            ) from e

        model_name = getattr(self.config, "mlx_model", self.DEFAULT_MODEL)
        logger.info("Loading MLX model: %s (first run downloads ~2 GB)", model_name)
        self._model, self._tokenizer = load(model_name)

    async def summarize(self, transcript: str) -> AsyncGenerator[str, None]:
        from mlx_lm import stream_generate

        prompt = self.config.prompt_template.format(transcript=transcript)
        if self.language_input != self.language_output:
            prompt += (
                f"\n\nIMPORTANT: The transcript is in {self.language_input}. "
                f"Please translate the resulting summary into {self.language_output}."
            )

        messages = [{"role": "user", "content": prompt}]
        formatted = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        for response in stream_generate(self._model, self._tokenizer, formatted, max_tokens=512):
            if response.text:
                yield response.text


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

class OllamaSummarizer(BaseSummarizer):
    """Uses Ollama Python SDK for streaming summarization."""

    def __init__(self, config: SummarizationConfig, language_input: str = "English", language_output: str = "English"):
        self.config = config
        self.language_input = language_input
        self.language_output = language_output
        self._ollama: Any = None
        self._load()

    def _load(self) -> None:
        try:
            import ollama
            self._ollama = ollama.AsyncClient(host=self.config.ollama_host)
        except ImportError as e:
            raise RuntimeError(
                "ollama package not installed. Run: uv pip install ollama"
            ) from e

    async def summarize(self, transcript: str) -> AsyncGenerator[str, None]:
        prompt = self.config.prompt_template.format(transcript=transcript)
        if self.language_input != self.language_output:
            prompt += (
                f"\n\nIMPORTANT: The transcript is in {self.language_input}. "
                f"Please translate the resulting summary into {self.language_output}."
            )

        stream = await self._ollama.chat(
            model=self.config.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseSummarizer]] = {
    "mlx": MLXSummarizer,
    "ollama": OllamaSummarizer,
}

if platform.system() == "Darwin":
    _REGISTRY["apple"] = AppleFoundationSummarizer


def create_summarizer(
    config: SummarizationConfig,
    language_input: str = "English",
    language_output: str = "English",
) -> BaseSummarizer:
    cls = _REGISTRY.get(config.backend)
    if cls is None:
        raise ValueError(f"Unknown summarization backend: {config.backend}")
    try:
        return cls(config, language_input, language_output)
    except RuntimeError as e:
        if config.backend != "apple":
            raise
        # Apple backend unavailable — fall back through MLX → Ollama without mutating config
        logger.warning("Apple backend unavailable (%s), trying MLX", e)
        try:
            return MLXSummarizer(config, language_input, language_output)
        except RuntimeError as e2:
            logger.warning("MLX backend unavailable (%s), falling back to Ollama", e2)
            return OllamaSummarizer(config, language_input, language_output)
