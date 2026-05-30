"""Transcription backend — Apple Speech, faster-whisper, or mlx-whisper."""
from __future__ import annotations

import platform
import threading
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional

import numpy as np

from .config import TranscriptionConfig
from note_assistant import logger


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseTranscriber(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe a chunk of float32 audio and return text."""

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Apple Speech backend (macOS 13+)
# ---------------------------------------------------------------------------

class AppleSpeechTranscriber(BaseTranscriber):
    """Uses Apple's Speech.framework via a subprocess worker.

    SFSpeechRecognizer callbacks cannot be delivered to daemon threads when
    asyncio occupies the main thread (asyncio's kqueue blocks CFRunLoop dispatch).
    Running recognition in a dedicated subprocess — which has no asyncio — avoids
    this entirely.  The persistent subprocess is started once and reused for all
    chunks, amortising startup cost.
    """

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self._lock = threading.Lock()
        self._proc: "subprocess.Popen | None" = None
        self._load()

    def _load(self) -> None:
        import subprocess
        import sys

        try:
            import Speech  # pyobjc-framework-Speech
            self._check_permission(Speech)
        except ImportError as e:
            raise RuntimeError(
                "pyobjc-framework-Speech not installed. "
                "Run: uv pip install pyobjc-framework-Speech"
            ) from e

        self._proc = subprocess.Popen(
            [sys.executable, "-m", "note_assistant._speech_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def _check_permission(Speech) -> None:  # noqa: N803
        status = Speech.SFSpeechRecognizer.authorizationStatus()
        if status == 3:  # Authorized
            return
        if status == 1:  # Denied
            raise RuntimeError(
                "Speech Recognition permission denied. "
                "Enable Note Assistant in System Settings › Privacy & Security › Speech Recognition."
            )
        if status == 2:  # Restricted
            raise RuntimeError("Speech Recognition is restricted on this device.")
        # NotDetermined — trigger dialog, then fail fast with instructions.
        Speech.SFSpeechRecognizer.requestAuthorization_(lambda _: None)
        raise RuntimeError(
            "Speech Recognition permission requested. "
            "Click Allow in the system dialog, then restart Note Assistant."
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        import struct

        if self._proc is None or self._proc.poll() is not None:
            logger.error("Speech worker process is not running")
            return ""

        rms = float(np.sqrt(np.mean(audio ** 2)))
        logger.debug("Transcribing chunk: samples=%d, RMS=%.6f", len(audio), rms)
        if rms < 1e-4:
            return ""

        audio_bytes = audio.astype(np.float32).tobytes()
        header = struct.pack(">II", len(audio_bytes), sample_rate)
        try:
            with self._lock:
                self._proc.stdin.write(header + audio_bytes)
                self._proc.stdin.flush()
                raw_len = self._proc.stdout.read(4)
                if len(raw_len) < 4:
                    return ""
                text_len = struct.unpack(">I", raw_len)[0]
                return self._proc.stdout.read(text_len).decode("utf-8")
        except Exception as e:
            logger.error("Speech worker IPC error: %s", e)
            return ""

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            self._proc.terminate()
        self._proc = None


# ---------------------------------------------------------------------------
# faster-whisper backend
# ---------------------------------------------------------------------------

class FasterWhisperTranscriber(BaseTranscriber):
    """Uses faster-whisper (CTranslate2) for transcription.

    Model loading is deferred to a background thread so the pipeline
    starts immediately. transcribe() skips chunks silently while the
    model is still initialising (which can take 10-60 s on first run).
    """

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self._model = None
        self._load_error: str | None = None
        self._error_emitted = False
        self._ready = threading.Event()
        # Fail fast if the package is missing — before spawning anything.
        try:
            import faster_whisper  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper not installed. Run: uv pip install faster-whisper"
            ) from e
        threading.Thread(target=self._load, daemon=True, name="whisper-load").start()

    def _load(self) -> None:
        try:
            from faster_whisper import WhisperModel
            device = self.config.device
            if device == "auto":
                device = "cpu"  # CTranslate2 auto-selects best available
            logger.info("Loading Whisper model '%s' on %s…", self.config.whisper_model, device)
            self._model = WhisperModel(
                self.config.whisper_model,
                device=device,
                compute_type="int8",
            )
            logger.info("Whisper model ready.")
        except Exception as e:
            self._load_error = str(e)
            logger.error("Failed to load Whisper model: %s", e, exc_info=True)
        finally:
            self._ready.set()

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if not self._ready.is_set():
            return ""  # model still loading — skip this chunk
        if self._load_error:
            # Surface the error once, then go silent to avoid log spam.
            if not self._error_emitted:
                self._error_emitted = True
                raise RuntimeError(f"Whisper model failed to load: {self._load_error}")
            return ""
        if self._model is None:
            return ""
        segments, _ = self._model.transcribe(
            audio,
            language=self.config.language,
            beam_size=5,
            vad_filter=True,
        )
        return " ".join(seg.text for seg in segments).strip()


# ---------------------------------------------------------------------------
# mlx-whisper backend (Apple Silicon — Metal/Neural Engine)
# ---------------------------------------------------------------------------

class MLXWhisperTranscriber(BaseTranscriber):
    """Uses mlx-whisper for transcription on Apple Silicon via Metal.

    Model download is deferred to a background thread so the pipeline
    starts immediately. transcribe() skips chunks silently while the
    model is still downloading (first run only).
    """

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self._mlx_whisper = None
        self._model_path: str | None = None
        self._load_error: str | None = None
        self._error_emitted = False
        self._ready = threading.Event()
        try:
            import mlx_whisper  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "mlx-whisper not installed. Run: uv pip install mlx-whisper"
            ) from e
        threading.Thread(target=self._load, daemon=True, name="mlx-whisper-load").start()

    def _load(self) -> None:
        try:
            import mlx_whisper
            from huggingface_hub import snapshot_download
            logger.info("Downloading/caching MLX Whisper model '%s'…", self.config.mlx_whisper_model)
            self._model_path = snapshot_download(repo_id=self.config.mlx_whisper_model)
            self._mlx_whisper = mlx_whisper
            logger.info("MLX Whisper model ready.")
        except Exception as e:
            self._load_error = str(e)
            logger.error("Failed to load MLX Whisper model: %s", e, exc_info=True)
        finally:
            self._ready.set()

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if not self._ready.is_set():
            return ""  # model still loading — skip this chunk
        if self._load_error:
            if not self._error_emitted:
                self._error_emitted = True
                raise RuntimeError(f"MLX Whisper model failed to load: {self._load_error}")
            return ""
        if self._mlx_whisper is None:
            return ""
        kwargs: dict = {}
        if self.config.language:
            kwargs["language"] = self.config.language
        result = self._mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model_path,
            verbose=False,
            **kwargs,
        )
        return result.get("text", "").strip()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseTranscriber]] = {
    "faster-whisper": FasterWhisperTranscriber,
}

if platform.system() == "Darwin":
    _REGISTRY["apple"] = AppleSpeechTranscriber

if platform.system() == "Darwin" and platform.machine() == "arm64":
    _REGISTRY["mlx-whisper"] = MLXWhisperTranscriber


def create_transcriber(config: TranscriptionConfig) -> BaseTranscriber:
    cls = _REGISTRY.get(config.backend)
    if cls is None:
        raise ValueError(f"Unknown transcription backend: {config.backend}")
    return cls(config)
