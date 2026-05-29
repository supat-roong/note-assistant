"""Main orchestrator — ties audio, transcription, summarization, notes together."""
from __future__ import annotations

import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from note_assistant import logger
from .audio_capture import AudioSource
from .config import AppConfig
from .errors import error_bus
from .notes_writer import NotesWriter
from .summarizer import BaseSummarizer, create_summarizer
from .transcriber import create_transcriber


class SummarizationWorker(threading.Thread):
    """Consumes transcript windows from a queue and summarizes them asynchronously."""

    MAX_QUEUE_DEPTH = 3

    def __init__(
        self,
        summarizer: BaseSummarizer,
        on_summary_token: Callable[[str], None],
        on_summary_complete: Callable[[str], None],
    ) -> None:
        super().__init__(daemon=True)
        self._summarizer = summarizer
        self._on_summary_token = on_summary_token
        self._on_summary_complete = on_summary_complete
        self._q: queue.Queue[str | None] = queue.Queue()
        self._paused_event = threading.Event()

    def enqueue(self, window: str) -> None:
        if self._q.qsize() >= self.MAX_QUEUE_DEPTH:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            logger.warning("summarization queue full — dropping window")
            error_bus.emit("summarizer", "Queue full — dropping oldest window", "warning")
        self._q.put(window)

    def pause(self) -> None:
        self._paused_event.set()

    def resume(self) -> None:
        self._paused_event.clear()

    def stop(self) -> None:
        self._q.put(None)

    def run(self) -> None:
        while True:
            window = self._q.get()
            if window is None:
                break
            if self._paused_event.is_set():
                continue
            try:
                tokens: list[str] = []
                for token in self._summarizer.summarize(window):
                    tokens.append(token)
                    self._on_summary_token(token)
                if tokens:
                    self._on_summary_complete("".join(tokens))
            except Exception as e:
                logger.error("Summarization error: %s", e)
                error_bus.emit("summarizer", str(e))


class NoteAssistantApp:
    """Orchestrates the full pipeline: audio → transcribe → summarize → notes."""

    def __init__(self, config: AppConfig, on_transcript: Callable[[str], None] | None = None,
                 on_summary: Callable[[str], None] | None = None,
                 on_chunk: Callable[[], None] | None = None):
        self.config = config
        self.on_transcript = on_transcript or (lambda x: None)
        self.on_summary = on_summary or (lambda x: None)
        self.on_chunk = on_chunk or (lambda: None)

        self._transcriber = create_transcriber(config.transcription)
        self._summarizer = create_summarizer(
            config.summarization,
            language_input=config.language_input,
            language_output=config.language_output
        )
        self._audio = AudioSource(config.audio)
        self._notes: NotesWriter | None = None
        self._running = False

        self._transcript_buffer: list[str] = []
        self._full_transcript: list[str] = []
        self._full_summary = ""
        self._chunk_count = 0
        self._start_time: datetime | None = None

        # Output directory
        if config.output.save_transcript or config.output.save_summary:
            config.output.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def elapsed(self) -> str:
        if self._start_time is None:
            return "00:00"
        delta = datetime.now() - self._start_time
        m, s = divmod(int(delta.total_seconds()), 60)
        return f"{m:02d}:{s:02d}"

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._running = True
        self._start_time = datetime.now()

        # Open Apple Notes session
        if self.config.output.apple_notes:
            self._notes = NotesWriter(self.config.output.apple_notes_title)
            self._notes.open_session()

        try:
            for audio_chunk in self._audio.stream():
                if not self._running:
                    break
                self.on_chunk()
                self._process_chunk(audio_chunk)
        finally:
            self._shutdown()

    def stop(self) -> None:
        self._running = False
        self._audio.stop()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _process_chunk(self, audio: np.ndarray) -> None:
        # Transcribe
        text = self._transcriber.transcribe(audio, self.config.audio.sample_rate)
        if not text:
            return

        self._chunk_count += 1
        self._transcript_buffer.append(text)
        self._full_transcript.append(text)
        self.on_transcript(text)

        if self._notes:
            self._notes.append_transcript(text + " ")

        # Summarize every N chunks
        if self._chunk_count % self.config.summarization.summarize_every == 0:
            self._trigger_summarization()

    def _trigger_summarization(self) -> None:
        window = " ".join(self._transcript_buffer)
        self._transcript_buffer.clear()

        summary_tokens: list[str] = []

        def _stream():
            for token in self._summarizer.summarize(window):
                summary_tokens.append(token)
                self.on_summary(token)

        t = threading.Thread(target=_stream, daemon=True)
        t.start()
        t.join(timeout=30)

        if summary_tokens:
            self._full_summary = "".join(summary_tokens)
            if self._notes:
                self._notes.update_summary(self._full_summary)

    def _shutdown(self) -> None:
        if self._notes:
            self._notes.close_session()

        cfg = self.config.output
        if not (cfg.save_transcript or cfg.save_summary):
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if cfg.save_transcript and self._full_transcript:
            p = cfg.output_dir / f"transcript_{ts}.txt"
            p.write_text("\n".join(self._full_transcript))

        if cfg.save_summary and self._full_summary:
            p = cfg.output_dir / f"summary_{ts}.txt"
            p.write_text(self._full_summary)
