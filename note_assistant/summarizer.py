"""Summarization backend — Apple Foundation Models, MLX (on-device), or Ollama."""
from __future__ import annotations

import platform
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

from .config import SummarizationConfig
from note_assistant import logger

# Known context window sizes (tokens) for specific models
_CONTEXT_LENGTHS: dict[str, int] = {
    "mlx-community/Qwen3-8B-4bit": 40_960,
    "mlx-community/Qwen3-14B-4bit": 131_072,
    "mlx-community/gemma-3-12b-it-4bit": 131_072,
    "mlx-community/gemma-3-12b-it-qat-4bit": 131_072,
    "mlx-community/gemma-4-e4b-it-4bit": 131_072,
    "gemma4:e4b": 131_072,
    "mlx-community/Qwen2.5-14B-Instruct-1M-4bit": 1_000_000,
    "mlx-community/Llama-3.2-3B-Instruct-4bit": 131_072,
    "qwen3:8b": 40_960,
    "qwen3:14b": 131_072,
    "gemma3:12b": 131_072,
    "gemma4:e4b": 131_072,
    "llama3.2:3b": 131_072,
    "llama3.1:8b": 131_072,
}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseSummarizer(ABC):
    TOKEN_LIMIT: int | None = None  # None = no practical limit

    @abstractmethod
    async def summarize(self, transcript: str) -> AsyncGenerator[str, None]:
        """Yield streaming summary tokens for the given transcript."""
        return
        yield  # pragma: no cover — marks this as an async generator

    async def generate_title(self, summary: str) -> str:
        """Generate a short title (≤5 words) from the summary. Override in subclasses."""
        return ""

    def warmup(self) -> None:
        """Pre-load model into memory so the first summarize() call is fast."""


# ---------------------------------------------------------------------------
# Apple Foundation Models backend
# Requires macOS 26+, Apple Silicon, Xcode 26, Apple Intelligence enabled.
# Install: uv add apple-fm-sdk
# ---------------------------------------------------------------------------

class AppleFoundationSummarizer(BaseSummarizer):
    """Uses Apple Foundation Models via apple-fm-sdk (requires macOS 26+, Apple Silicon)."""
    TOKEN_LIMIT = 4096

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
        # Try requested output language first, fall back to English if unsupported
        for lang in [self.language_output, "English"]:
            prompt = (
                f"Generate a concise, informative title of 5 words or less for these notes. "
                f"Write the title in {lang}. "
                "Reply with ONLY the title — no quotes, no punctuation at the end:\n\n"
                + summary[:1000]
            )
            result = ""
            try:
                session = self._fm.LanguageModelSession()
                async for chunk in session.stream_response(prompt):
                    result = chunk
                if result.strip():
                    return result.strip().strip('"').strip("'").rstrip(".")
            except Exception as e:
                logger.debug("generate_title in %s failed: %s", lang, e)
        return ""


# ---------------------------------------------------------------------------
# MLX backend — on-device LLM via Apple Silicon GPU/Neural Engine (Python-native)
# ---------------------------------------------------------------------------

class MLXSummarizer(BaseSummarizer):
    """On-device summarization using MLX (Apple Silicon GPU). Lazy-loads model on first use."""

    DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"

    def __init__(self, config: SummarizationConfig, language_input: str = "English",
                 language_output: str = "English", model_override: str | None = None):
        self.config = config
        self.language_input = language_input
        self.language_output = language_output
        self._model_name = model_override or getattr(config, "mlx_model", self.DEFAULT_MODEL)
        self.TOKEN_LIMIT = _CONTEXT_LENGTHS.get(self._model_name, 40_960)
        self._model = None
        self._tokenizer = None
        # Validate mlx-lm is installed at init time (fail fast)
        try:
            import mlx_lm  # noqa: F401
        except ImportError as e:
            raise RuntimeError("mlx-lm not installed. Run: uv add mlx mlx-lm") from e

    def _load(self) -> None:
        from mlx_lm import load
        logger.info("Loading MLX model: %s", self._model_name)
        self._model, self._tokenizer = load(self._model_name)

    def warmup(self) -> None:
        if self._model is None:
            self._load()

    async def summarize(self, transcript: str) -> AsyncGenerator[str, None]:
        from mlx_lm import stream_generate
        if self._model is None:
            self._load()

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

    def __init__(self, config: SummarizationConfig, language_input: str = "English",
                 language_output: str = "English", model_override: str | None = None):
        self.config = config
        self.language_input = language_input
        self.language_output = language_output
        self._model_name = model_override or config.ollama_model
        self.TOKEN_LIMIT = _CONTEXT_LENGTHS.get(self._model_name, 40_960)
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

    def warmup(self) -> None:
        import asyncio
        async def _ping() -> None:
            try:
                await self._ollama.chat(
                    model=self._model_name,
                    messages=[{"role": "user", "content": "hi"}],
                    stream=False,
                    options={"num_predict": 1},
                )
            except Exception:
                pass
        asyncio.run(_ping())

    async def summarize(self, transcript: str) -> AsyncGenerator[str, None]:
        prompt = self.config.prompt_template.format(transcript=transcript)
        if self.language_input != self.language_output:
            prompt += (
                f"\n\nIMPORTANT: The transcript is in {self.language_input}. "
                f"Please translate the resulting summary into {self.language_output}."
            )

        stream = await self._ollama.chat(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token

    async def generate_title(self, summary: str) -> str:
        prompt = (
            f"Generate a concise, informative title of 5 words or less for these notes. "
            f"Write the title in {self.language_output}. "
            "Reply with ONLY the title — no quotes, no punctuation at the end:\n\n"
            + summary[:1000]
        )
        response = await self._ollama.chat(
            model=self._model_name,
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
