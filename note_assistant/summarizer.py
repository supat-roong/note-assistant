"""Summarization backend — Apple Foundation Models, MLX (on-device), or Ollama."""
from __future__ import annotations

import platform
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Iterator

from .config import SummarizationConfig
from note_assistant import logger


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseSummarizer(ABC):
    @abstractmethod
    def summarize(self, transcript: str) -> Iterator[str]:
        """Yield streaming summary tokens for the given transcript."""
        yield from ()


# ---------------------------------------------------------------------------
# Apple Foundation Models backend
# Requires macOS 26+ SDK (Xcode 26). Uses FoundationModels framework.
# Falls back automatically when SDK is unavailable.
# ---------------------------------------------------------------------------

# macOS 26+ renamed Intelligence → FoundationModels
FOUNDATION_MODEL_SWIFT = """
import Foundation
import FoundationModels

guard #available(macOS 26.0, *) else {
    print("ERROR: FoundationModels requires macOS 26.0+")
    exit(1)
}

let prompt = CommandLine.arguments.dropFirst().joined(separator: " ")

let model = SystemLanguageModel.default
guard model.availability == .available else {
    print("ERROR: Apple Intelligence not available/enabled on this device")
    exit(1)
}

let session = LanguageModelSession()
Task {
    do {
        let stream = session.streamResponse(to: prompt)
        for try await token in stream {
            print(token, terminator: "")
            fflush(stdout)
        }
        exit(0)
    } catch {
        print("ERROR: \\(error.localizedDescription)")
        exit(1)
    }
}
RunLoop.main.run(until: .distantFuture)
"""


class AppleFoundationSummarizer(BaseSummarizer):
    """Uses Apple Foundation Models via a compiled Swift helper (requires Xcode 26 SDK)."""

    def __init__(self, config: SummarizationConfig, language_input: str = "English", language_output: str = "English"):
        self.config = config
        self.language_input = language_input
        self.language_output = language_output
        self._helper_path = self._ensure_helper()

    def _ensure_helper(self) -> str:
        from pathlib import Path

        helper_dir = Path.home() / ".note-assistant"
        helper_dir.mkdir(exist_ok=True)
        helper_path = helper_dir / "foundation_model_bridge"
        swift_src = helper_dir / "foundation_model_bridge.swift"

        if not helper_path.exists():
            swift_src.write_text(FOUNDATION_MODEL_SWIFT)
            result = subprocess.run(
                ["swiftc", str(swift_src), "-o", str(helper_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "FoundationModels Swift bridge failed to compile — "
                    "requires Xcode 26 SDK (macOS 26+). "
                    "Install Xcode from the App Store to enable on-device Apple AI."
                )
        return str(helper_path)

    def summarize(self, transcript: str) -> Iterator[str]:
        prompt = self.config.prompt_template.format(transcript=transcript)
        if self.language_input != self.language_output:
            prompt += (
                f"\n\nIMPORTANT: The transcript is in {self.language_input}. "
                f"Please translate the resulting summary into {self.language_output}."
            )

        proc = subprocess.Popen(
            [self._helper_path, prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        if proc.stdout:
            for line in proc.stdout:
                yield line
        proc.wait()


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
        from mlx_lm import load
        self._model, self._tokenizer = load(model_name)

    def summarize(self, transcript: str) -> Iterator[str]:
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
            token = response.text
            if token:
                yield token


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
            self._ollama = ollama
        except ImportError as e:
            raise RuntimeError(
                "ollama package not installed. Run: uv pip install ollama"
            ) from e

    def summarize(self, transcript: str) -> Iterator[str]:
        prompt = self.config.prompt_template.format(transcript=transcript)
        if self.language_input != self.language_output:
            prompt += (
                f"\n\nIMPORTANT: The transcript is in {self.language_input}. "
                f"Please translate the resulting summary into {self.language_output}."
            )

        stream = self._ollama.chat(
            model=self.config.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"host": self.config.ollama_host},
        )
        for chunk in stream:
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
