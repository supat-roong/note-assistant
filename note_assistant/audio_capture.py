"""Audio capture — microphone or system audio via sounddevice."""
from __future__ import annotations

import queue
import threading
from typing import Generator, Literal

import librosa
import numpy as np
import sounddevice as sd

from .config import AudioConfig
from note_assistant import logger


def list_devices() -> None:
    """Print all available audio input devices."""
    devices = sd.query_devices()
    print("\nAvailable audio input devices:")
    print("-" * 50)
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = " ← default" if i == sd.default.device[0] else ""
            print(f"  [{i:2d}] {dev['name']}{marker}")
    print()


def find_blackhole_device() -> int | None:
    """Find BlackHole virtual audio device index."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if "BlackHole" in dev["name"] and dev["max_input_channels"] > 0:
            return i
    return None


class AudioSource:
    """Streams audio chunks from mic or system audio (via BlackHole)."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.sample_rate = config.sample_rate
        self.chunk_frames = int(config.sample_rate * config.chunk_seconds)
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stop_event = threading.Event()
        self._stream: sd.InputStream | None = None
        self._total_chunks: int = 0

    @property
    def total_chunks(self) -> int:
        return self._total_chunks

    def _resolve_device(self) -> int | None:
        if self.config.source == "mic":
            return None  # sounddevice default mic
        # system audio via BlackHole
        dev = find_blackhole_device()
        if dev is None:
            raise RuntimeError(
                "BlackHole audio device not found.\n"
                "Run: brew install blackhole-2ch\n"
                "Then create an Aggregate Device in Audio MIDI Setup."
            )
        return dev

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:  # noqa: ANN001
        if status:
            logger.debug("Audio status: %s", status)

        self._q.put(indata.copy().flatten())

    def stream(self) -> Generator[np.ndarray, None, None]:
        """Yield numpy float32 audio chunks of `chunk_frames` samples."""
        if self.config.source == "file":
            yield from self._stream_file()
        else:
            yield from self._stream_live()

    def _stream_file(self) -> Generator[np.ndarray, None, None]:
        if not self.config.file_path or not self.config.file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.config.file_path}")
        
        # Load and resample to 16k
        # Note: sr=self.sample_rate ensures it matches what the transcribers expect
        logger.debug("Loading audio file: %s", self.config.file_path)
        y, _ = librosa.load(str(self.config.file_path), sr=self.sample_rate)
        self._total_chunks = max(1, -(-len(y) // self.chunk_frames))  # ceiling division

        for i in range(0, len(y), self.chunk_frames):
            if self._stop_event.is_set():
                break
            chunk = y[i : i + self.chunk_frames]
            if len(chunk) < self.chunk_frames:
                chunk = np.pad(chunk, (0, self.chunk_frames - len(chunk)))
            yield chunk

    def _stream_live(self) -> Generator[np.ndarray, None, None]:
        device = self._resolve_device()
        buffer = np.array([], dtype=np.float32)

        self._stream = sd.InputStream(
            device=device,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=self._callback,
        )
        with self._stream:
            while not self._stop_event.is_set():
                try:
                    chunk = self._q.get(timeout=0.5)
                    buffer = np.concatenate([buffer, chunk])
                    while len(buffer) >= self.chunk_frames:
                        yield buffer[: self.chunk_frames]
                        buffer = buffer[self.chunk_frames :]
                except queue.Empty:
                    continue

    def stop(self) -> None:
        self._stop_event.set()
