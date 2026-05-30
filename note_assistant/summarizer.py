"""Summarization backend — Apple Foundation Models, MLX (on-device), or Ollama."""
from __future__ import annotations

import platform
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

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

    async def generate_title(self, summary: str) -> str:
        """Generate a short title (≤5 words) from the summary. Override in subclasses."""
        return ""


# ---------------------------------------------------------------------------
# Apple Foundation Models backend
# Requires macOS 26+, Apple Silicon, Xcode 26, Apple Intelligence enabled.
# Install: uv add apple-fm-sdk
# ---------------------------------------------------------------------------

class AppleFoundationSummarizer(BaseSummarizer):
    """Uses Apple Foundation Models via apple-fm-sdk (requires macOS 26+, Apple Silicon)."""

    def __init__(
        self,
        config: SummarizationConfig,
        language_input: str = "English",
        language_output: str = "English",
    ) -> None:
        self.config = config
        self.language_input = language_input
        self.language_output = language_output
        self._check_availability()

    def _check_availability(self) -> None:
        try:
            import apple_fm_sdk as fm
            self._fm = fm
        except ImportError as e:
            raise RuntimeError(
                "apple-fm-sdk not installed. Run: uv add apple-fm-sdk"
            ) from e
        model = fm.SystemLanguageModel()
        available, reason = model.is_available()
        if not available:
            raise RuntimeError(f"Apple Foundation Models not available: {reason}")

    async def summarize(self, transcript: str) -> AsyncGenerator[str, None]:
        fm = self._fm
        prompt = self.config.prompt_template.format(transcript=transcript)
        if self.language_input != self.language_output:
            prompt += (
                f"\n\nIMPORTANT: The transcript is in {self.language_input}. "
                f"Please translate the resulting summary into {self.language_output}."
            )
        session = fm.LanguageModelSession()
        try:
            async for chunk in session.stream_response(prompt):
                yield chunk
        except fm.ExceededContextWindowSizeError:
            logger.warning(
                "Transcript exceeded 4096-token context window — retrying with truncated transcript"
            )
            # Truncate the transcript (not the formatted prompt) so the instruction header is preserved
            overhead = len(prompt) - len(transcript)
            budget = max(0, 3000 - overhead)
            retry_prompt = self.config.prompt_template.format(transcript=transcript[-budget:])
            if self.language_input != self.language_output:
                retry_prompt += (
                    f"\n\nIMPORTANT: The transcript is in {self.language_input}. "
                    f"Please translate the resulting summary into {self.language_output}."
                )
            retry_session = fm.LanguageModelSession()
            async for chunk in retry_session.stream_response(retry_prompt):
                yield chunk
        except fm.AssetsUnavailableError as e:
            raise RuntimeError(f"Apple Intelligence assets unavailable: {e}") from e

    async def generate_title(self, summary: str) -> str:
        prompt = (
            "Generate a concise, informative title of 5 words or less for these notes. "
            "Reply with ONLY the title — no quotes, no punctuation at the end:\n\n"
            + summary[:1000]
        )
        session = self._fm.LanguageModelSession()
        result = ""
        try:
            async for chunk in session.stream_response(prompt):
                result = chunk
        except Exception:
            pass
        return result.strip().strip('"').strip("'").rstrip(".")


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

    async def generate_title(self, summary: str) -> str:
        prompt = (
            "Generate a concise, informative title of 5 words or less for these notes. "
            "Reply with ONLY the title — no quotes, no punctuation at the end:\n\n"
            + summary[:1000]
        )
        response = await self._ollama.chat(
            model=self.config.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return response["message"]["content"].strip().strip('"').strip("'").rstrip(".")


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
