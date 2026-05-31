"""Summarization backend — Apple Foundation Models, MLX (on-device), or Ollama."""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import platform
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

from .config import SummarizationConfig
from note_assistant import logger


@contextlib.contextmanager
def _suppress_c_stderr():
    """Redirect fd 2 to /dev/null briefly to silence C-level macOS malloc noise."""
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_fd = os.dup(2)
        os.dup2(devnull_fd, 2)
        try:
            yield
        finally:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
            os.close(devnull_fd)
    except OSError:
        yield


def _mlx_context_length(model_name: str) -> int | None:
    """Read context window from the model's cached HuggingFace config.json."""
    slug = model_name.replace("/", "--").replace(":", "-")
    base = pathlib.Path.home() / ".cache/huggingface/hub"
    configs = sorted(base.glob(f"models--{slug}/snapshots/*/config.json"))
    if not configs:
        return None
    try:
        data = json.loads(configs[-1].read_text())
        # Collect all dicts to search (top-level + one level of nesting)
        sections = [data] + [v for v in data.values() if isinstance(v, dict)]
        # 1. Explicit keys
        for section in sections:
            for key in ("max_position_embeddings", "max_context_length", "max_seq_len"):
                val = section.get(key)
                if val and isinstance(val, int) and 1024 <= val <= 10_000_000:
                    return val
        # 2. rope_scaling: base × factor when original_max_position_embeddings is set
        for section in sections:
            rs = section.get("rope_scaling")
            if isinstance(rs, dict):
                orig = rs.get("original_max_position_embeddings")
                factor = rs.get("factor", 1)
                if orig:
                    return int(orig * factor)
    except Exception:
        pass
    # 3. Family fallback for models whose configs omit the context window
    _FAMILY_CONTEXT = {
        "gemma-3": 131_072, "gemma3": 131_072,
        "gemma-4": 131_072, "gemma4": 131_072,
        "qwen2.5": 131_072, "qwen3": 40_960,
        "llama-3.1": 131_072, "llama-3.2": 131_072,
    }
    lower = model_name.lower()
    for family, ctx in _FAMILY_CONTEXT.items():
        if family in lower:
            return ctx
    return None


def _ollama_context_length(model_name: str, host: str = "http://localhost:11434") -> int | None:
    """Query Ollama's /api/show endpoint for the model's context length."""
    try:
        import urllib.request, json as _json
        req = urllib.request.Request(
            f"{host}/api/show",
            data=_json.dumps({"name": model_name}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            info = _json.loads(resp.read())
        mi = info.get("model_info", {})
        for key, val in mi.items():
            if key.endswith(".context_length"):
                return int(val)
    except Exception:
        pass
    return None


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

    async def generate_title(self, summary: str, title_prompt_template: str) -> str:
        """Generate a short title (≤5 words) from the summary. Override in subclasses."""
        return ""

    def warmup(self) -> None:
        """Pre-load model into memory so the first summarize() call is fast."""

    def close(self) -> None:
        """Release any resources held by this summarizer. No-op by default."""

    @property
    def model_label(self) -> str:
        """Human-readable model name for display in the UI."""
        return ""


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
        with _suppress_c_stderr():
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

    @property
    def model_label(self) -> str:
        return "Apple Intelligence"

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

    async def generate_title(self, summary: str, title_prompt_template: str) -> str:
        for lang in [self.language_output, "English"]:
            prompt = title_prompt_template.format(language=lang, summary=summary[:1000])
            result = ""
            try:
                session = self._fm.LanguageModelSession()
                async for chunk in session.stream_response(prompt):
                    result += chunk
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
        self.TOKEN_LIMIT = _mlx_context_length(self._model_name) or 40_960
        logger.debug("MLX %s context length: %d", self._model_name, self.TOKEN_LIMIT)
        self._model = None
        self._tokenizer = None
        # Validate mlx-lm is installed at init time (fail fast)
        try:
            import mlx_lm  # noqa: F401
        except ImportError as e:
            raise RuntimeError("mlx-lm not installed. Run: uv add mlx mlx-lm") from e

    @property
    def model_label(self) -> str:
        return f"{self._model_name.split('/')[-1]} (mlx)"

    def _load(self) -> None:
        from mlx_lm import load
        logger.info("Loading MLX model: %s", self._model_name)
        self._model, self._tokenizer = load(self._model_name)

    def warmup(self) -> None:
        if self._model is None:
            self._load()

    def close(self) -> None:
        """Unload model weights from GPU/Metal memory."""
        if self._model is not None:
            logger.info("Unloading MLX model from memory: %s", self._model_name)
            self._model = None
            self._tokenizer = None
            try:
                import mlx.core as mx
                mx.metal.clear_cache()
            except Exception:
                pass

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
        try:
            # enable_thinking=False suppresses Qwen3's chain-of-thought block
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
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
        self._owned_process: subprocess.Popen | None = None
        self._ensure_ollama_running()
        self.TOKEN_LIMIT = _ollama_context_length(self._model_name, config.ollama_host) or 40_960
        logger.debug("Ollama %s context length: %d", self._model_name, self.TOKEN_LIMIT)
        self._ollama: Any = None
        self._load()

    def _ensure_ollama_running(self) -> None:
        """Start Ollama server if not reachable; record process ownership."""
        import urllib.request
        url = f"{self.config.ollama_host}/api/tags"
        try:
            urllib.request.urlopen(url, timeout=2)
            return  # already running — don't own it
        except Exception:
            pass

        try:
            self._owned_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            logger.error("ollama binary not found in PATH — cannot auto-start Ollama server")
            return

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(url, timeout=1)
                logger.info("Ollama server started (pid %d)", self._owned_process.pid)
                return
            except Exception:
                time.sleep(0.5)
        logger.warning("Ollama server did not become reachable within 10 seconds")

    def close(self) -> None:
        """Terminate the Ollama process if we started it."""
        if self._owned_process is None:
            return
        self._owned_process.terminate()
        try:
            self._owned_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._owned_process.kill()
        self._owned_process = None

    @property
    def model_label(self) -> str:
        return f"{self._model_name} (ollama)"

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
            think=False,
        )
        async for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token

    async def generate_title(self, summary: str, title_prompt_template: str) -> str:
        prompt = title_prompt_template.format(language=self.language_output, summary=summary[:1000])
        response = await self._ollama.chat(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            think=False,
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
