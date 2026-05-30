"""Transcription backend — Apple Speech (default) or faster-whisper."""
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
    """Uses Apple's Speech.framework via pyobjc for on-device transcription."""

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        try:
            import Speech  # pyobjc-framework-Speech
            self._Speech = Speech
            self._check_permission()
        except ImportError as e:
            raise RuntimeError(
                "pyobjc-framework-Speech not installed. "
                "Run: uv pip install pyobjc-framework-Speech"
            ) from e

    def _check_permission(self) -> None:
        status = self._Speech.SFSpeechRecognizer.authorizationStatus()
        statuses = {
            0: "NotDetermined",
            1: "Denied",
            2: "Restricted",
            3: "Authorized",
        }
        logger.debug("Apple Speech authorization: %s", statuses.get(status, "Unknown"))
        if status == 0:  # NotDetermined
            logger.info("Requesting Speech authorization")
            self._Speech.SFSpeechRecognizer.requestAuthorization_(lambda _: None)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio array using Apple Speech framework."""
        import AVFoundation
        import Foundation
        import Speech

        # Log audio level for debugging
        rms = np.float64(np.sqrt(np.mean(audio**2)))
        logger.debug("Transcribing chunk: samples=%d, RMS=%.6f", len(audio), rms)

        if rms < 1e-4:
            # Silence threshold
            return ""

        # Convert numpy float32 → AVAudioPCMBuffer
        fmt = AVFoundation.AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
            AVFoundation.AVAudioPCMFormatFloat32, sample_rate, 1, True
        )
        capacity = len(audio)
        buf = AVFoundation.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(fmt, capacity)
        buf.setFrameLength_(capacity)

        # Copy numpy data into the buffer's float channel data.
        # objc.varlist (returned by floatChannelData) is not accepted by ctypes.memmove;
        # use as_buffer() + memoryview byte copy instead.
        src_bytes = audio.tobytes()
        channel_data = buf.floatChannelData()[0]
        memoryview(channel_data.as_buffer(len(src_bytes))).cast("B")[:len(src_bytes)] = memoryview(src_bytes).cast("B")

        recognizer = Speech.SFSpeechRecognizer.alloc().initWithLocale_(
            Foundation.NSLocale.currentLocale()
        )
        if not recognizer:
            return ""

        recognizer.setSupportsOnDeviceRecognition_(True)
        if not recognizer.isAvailable():
            logger.error("SFSpeechRecognizer not available — check System Settings")
            return ""

        request = Speech.SFSpeechAudioBufferRecognitionRequest.alloc().init()
        if not request:
            return ""
        
        request.appendAudioPCMBuffer_(buf)
        request.endAudio()

        result_text = ""
        done = threading.Event()

        def handler(result, error):  # noqa: ANN001
            nonlocal result_text
            if error:
                logger.error("Recognition error: %s", error.localizedDescription())
            if result:
                result_text = result.bestTranscription().formattedString()
            if result and result.isFinal():
                done.set()
            elif not result:
                done.set()

        task = recognizer.recognitionTaskWithRequest_resultHandler_(request, handler)
        # Pump the RunLoop so Speech framework callbacks can fire on this thread.
        # done.wait() alone doesn't work because Speech callbacks require an active RunLoop.
        import time
        from CoreFoundation import CFRunLoopRunInMode, kCFRunLoopDefaultMode
        deadline = time.monotonic() + 5.0
        while not done.is_set() and time.monotonic() < deadline:
            CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.1, True)
        if not done.is_set():
            logger.debug("Recognition timed out for chunk")

        return result_text.strip()


# ---------------------------------------------------------------------------
# faster-whisper backend
# ---------------------------------------------------------------------------

class FasterWhisperTranscriber(BaseTranscriber):
    """Uses faster-whisper (CTranslate2) for transcription."""

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self._model = None
        self._load()

    def _load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper not installed. Run: uv pip install faster-whisper"
            ) from e

        device = self.config.device
        if device == "auto":
            device = "cpu"  # CTranslate2 auto-selects best available

        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            self.config.whisper_model,
            device=device,
            compute_type="int8",
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
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
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseTranscriber]] = {
    "faster-whisper": FasterWhisperTranscriber,
}

if platform.system() == "Darwin":
    _REGISTRY["apple"] = AppleSpeechTranscriber


def create_transcriber(config: TranscriptionConfig) -> BaseTranscriber:
    cls = _REGISTRY.get(config.backend)
    if cls is None:
        raise ValueError(f"Unknown transcription backend: {config.backend}")
    return cls(config)
