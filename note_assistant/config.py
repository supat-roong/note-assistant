"""Configuration models — loaded from config.yaml, merged with CLI overrides."""
from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class AudioConfig(BaseModel):
    source: Literal["mic", "system", "file"] = "mic"
    file_path: Optional[Path] = None
    sample_rate: int = 16_000
    chunk_seconds: float = 2.0


class TranscriptionConfig(BaseModel):
    backend: Literal["apple", "faster-whisper", "mlx-whisper"] = "apple"
    language: Optional[str] = None
    whisper_model: str = "base"
    mlx_whisper_model: str = "mlx-community/whisper-base-mlx"
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"


class SummarizationConfig(BaseModel):
    backend: Literal["apple", "mlx", "ollama"] = "apple"
    summarize_every: int = Field(3, ge=1)
    prompt_template: str = (
        "Summarize the following transcript into concise bullet-point notes.\n"
        "Write each bullet point on its own line starting with '- '.\n\n"
        "{transcript}"
    )
    mlx_model: str = "mlx-community/Qwen3-8B-4bit"
    mlx_fallback_model: str = "mlx-community/gemma-4-e4b-it-OptiQ-4bit"
    ollama_model: str = "qwen3:8b"
    ollama_fallback_model: str = "qwen3:4b"
    ollama_host: str = "http://localhost:11434"


class OutputConfig(BaseModel):
    save_transcript: bool = True
    save_summary: bool = True
    output_dir: Path = Path("./notes")
    apple_notes: bool = True
    apple_notes_title: str = "Note Assistant — {date}"
    auto_title: bool = True
    title_prompt_template: str = (
        "Generate a concise, informative title of 5 words or less for these notes. "
        "Write the title in {language}. "
        "Reply with ONLY the title — no quotes, no punctuation at the end:\n\n"
        "{summary}"
    )
    save_recording: bool = False
    recording_dir: Optional[Path] = None

    @field_validator("recording_dir", mode="before")
    @classmethod
    def _expand_recording_dir(cls, v):
        if v is not None:
            return Path(v).expanduser()
        return v


class AppConfig(BaseModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    language_input: str = "English"
    language_output: str = "English"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: Path | str | None = None, **overrides) -> AppConfig:
    """Load config from YAML file, then apply dict overrides."""
    data: dict = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            with p.open() as fh:
                data = yaml.safe_load(fh) or {}
    config = AppConfig.model_validate(data)
    # Apply flat CLI overrides (e.g. source="system", whisper_model="small")
    for key, value in overrides.items():
        if value is None:
            continue
        if key == "source":
            config.audio.source = value  # type: ignore[assignment]
        elif key == "transcription_backend":
            config.transcription.backend = value  # type: ignore[assignment]
        elif key == "whisper_model":
            config.transcription.whisper_model = value
        elif key == "summarization_backend":
            config.summarization.backend = value  # type: ignore[assignment]
        elif key == "ollama_model":
            config.summarization.ollama_model = value
        elif key == "chunk_seconds":
            config.audio.chunk_seconds = value
        elif key == "log_level":
            config.log_level = value
    return config


# ---------------------------------------------------------------------------
# Capability detection (also used by setup_mac.sh equivalent in Python)
# ---------------------------------------------------------------------------

def detect_best_backends() -> tuple[str, str]:
    """Return (transcription_backend, summarization_backend) for this machine."""
    if platform.system() != "Darwin":
        return "faster-whisper", "ollama"

    # macOS version
    ver_str = platform.mac_ver()[0]  # e.g. "15.1"
    try:
        major, minor = (int(x) for x in ver_str.split(".")[:2])
    except ValueError:
        return "faster-whisper", "ollama"

    # Chip: arm64 = Apple Silicon
    chip = subprocess.run(["uname", "-m"], capture_output=True, text=True).stdout.strip()
    is_apple_silicon = chip == "arm64"

    if major >= 26:
        if is_apple_silicon:
            return "apple", "apple"
        else:
            return "apple", "ollama"
    elif major >= 13:
        return "apple", "ollama"
    else:
        return "faster-whisper", "ollama"
