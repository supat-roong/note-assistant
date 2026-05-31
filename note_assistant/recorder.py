"""Session audio recorder — streams to WAV, encodes to MP3 + M4A at shutdown."""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from note_assistant import logger


class SessionRecorder:
    """Streams float32 audio chunks to a temp WAV, then encodes to MP3 + M4A via ffmpeg."""

    def __init__(self, recording_dir: Path, sample_rate: int = 16_000):
        self._recording_dir = recording_dir
        self._sample_rate = sample_rate
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._wav_path = recording_dir / f"recording_{ts}.wav"
        self._m4a_path = recording_dir / f"recording_{ts}_tmp.m4a"
        self._mp3_path = recording_dir / f"recording_{ts}.mp3"
        self._sf: sf.SoundFile | None = None
        self._started: bool = False
        self._finished: bool = False

    def start(self) -> None:
        self._recording_dir.mkdir(parents=True, exist_ok=True)
        self._sf = sf.SoundFile(
            str(self._wav_path),
            mode="w",
            samplerate=self._sample_rate,
            channels=1,
            subtype="FLOAT",
        )
        self._started = True

    def write(self, chunk: np.ndarray) -> None:
        if self._sf is None:
            return
        self._sf.write(chunk)

    def finish(self) -> tuple[Path, Path]:
        """Close WAV and encode to MP3 + M4A. Returns (mp3_path, m4a_path).

        Raises RuntimeError if called before start() or called a second time.
        Raises FileNotFoundError if ffmpeg is not installed.
        Raises subprocess.CalledProcessError if ffmpeg encoding fails.
        """
        if not self._started:
            raise RuntimeError("SessionRecorder.finish() called before start()")
        if self._finished:
            raise RuntimeError("SessionRecorder.finish() already called")

        if self._sf is not None:
            self._sf.close()
            self._sf = None

        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(self._wav_path),
                "-ar", str(self._sample_rate), "-ac", "1", "-q:a", "4",
                str(self._mp3_path),
            ],
            capture_output=True,
            check=True,
        )
        logger.debug("Encoded recording to MP3: %s", self._mp3_path)

        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(self._wav_path),
                "-ar", str(self._sample_rate), "-ac", "1", "-c:a", "aac",
                str(self._m4a_path),
            ],
            capture_output=True,
            check=True,
        )
        logger.debug("Encoded recording to M4A: %s", self._m4a_path)

        self._finished = True
        return self._mp3_path, self._m4a_path

    def cleanup(self) -> None:
        for path in (self._wav_path, self._m4a_path):
            try:
                path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Could not remove temp file %s: %s", path, e)
