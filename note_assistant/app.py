"""Main orchestrator — ties audio, transcription, summarization, notes together."""
from __future__ import annotations

import asyncio
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
from .transcriber import BaseTranscriber, create_transcriber


class SummarizationWorker(threading.Thread):
    """Consumes transcript windows from a queue and summarizes them asynchronously."""

    MAX_QUEUE_DEPTH = 3

    def __init__(
        self,
        summarizer: BaseSummarizer,
        on_summary_token: Callable[[str], None],
        on_summary_complete: Callable[[str], None],
        on_summary_start: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self._summarizer = summarizer
        self._on_summary_token = on_summary_token
        self._on_summary_complete = on_summary_complete
        self._on_summary_start = on_summary_start or (lambda: None)
        self._q: queue.Queue[str | None] = queue.Queue()
        self._paused_event = threading.Event()
        self._last_queue_warning = 0.0

    def enqueue(self, window: str) -> None:
        import time
        if self._q.qsize() >= self.MAX_QUEUE_DEPTH:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            logger.warning("summarization queue full — dropping window")
            now = time.monotonic()
            if now - self._last_queue_warning > 10.0:
                self._last_queue_warning = now
                error_bus.emit("summarizer", "Queue full — dropping oldest window", "warning")
        self._q.put(window)

    def pause(self) -> None:
        self._paused_event.set()

    def resume(self) -> None:
        self._paused_event.clear()

    def stop(self) -> None:
        self._q.put(None)

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while True:
                window = self._q.get()
                if window is None:
                    break
                if self._paused_event.is_set():
                    continue
                try:
                    loop.run_until_complete(self._process_async(window))
                except Exception as e:
                    logger.error("Summarization error: %s", e)
                    error_bus.emit("summarizer", str(e))
        finally:
            loop.close()

    async def _process_async(self, window: str) -> None:
        last = ""
        async for chunk in self._summarizer.summarize(window):
            if not last:
                self._on_summary_start()
            last = chunk
            self._on_summary_token(chunk)  # cumulative — UI must replace, not append
        if last:
            self._on_summary_complete(last)


class NoteAssistantApp:
    """Orchestrates the full pipeline: audio → transcribe → summarize → notes."""

    def __init__(
        self,
        config: AppConfig,
        on_transcript: Callable[[str], None] | None = None,
        on_summary: Callable[[str], None] | None = None,
        on_summary_start: Callable[[], None] | None = None,
        on_chunk: Callable[[], None] | None = None,
        on_error: Callable[[str, str, str], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        transcriber: "BaseTranscriber | None" = None,
        summarizer: "BaseSummarizer | None" = None,
    ):
        self.config = config
        self.on_transcript = on_transcript or (lambda x: None)
        self.on_summary = on_summary or (lambda x: None)
        self.on_summary_start = on_summary_start or (lambda: None)
        self.on_chunk = on_chunk or (lambda: None)
        self.on_error = on_error or (lambda s, m, sev: None)
        self.on_progress = on_progress or (lambda cur, tot: None)

        self._transcriber = transcriber or create_transcriber(config.transcription)
        self._summarizer = summarizer or create_summarizer(
            config.summarization,
            language_input=config.language_input,
            language_output=config.language_output,
        )
        self._audio = AudioSource(config.audio)
        self._notes: NotesWriter | None = None
        self._running = False
        self._paused_event = threading.Event()

        self._since_last_summary: list[str] = []
        self._full_transcript: list[str] = []
        self._full_summary = ""
        self._chunk_count = 0
        self._start_time: datetime | None = None

        self._worker = SummarizationWorker(
            self._summarizer,
            on_summary_token=self.on_summary,
            on_summary_complete=self._on_summary_complete,
            on_summary_start=self.on_summary_start,
        )

        error_bus.subscribe(self._route_error)

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
    # Public control
    # ------------------------------------------------------------------

    def pause(self) -> None:
        self._paused_event.set()
        self._worker.pause()

    def resume(self) -> None:
        self._paused_event.clear()
        self._worker.resume()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._running = True
        self._start_time = datetime.now()
        self._worker.start()

        if self.config.output.apple_notes:
            self._notes = NotesWriter(self.config.output.apple_notes_title)
            self._notes.open_session()

        try:
            audio_chunk_idx = 0
            for audio_chunk in self._audio.stream():
                if not self._running:
                    break
                audio_chunk_idx += 1
                self.on_chunk()
                self._process_chunk(audio_chunk)
                if self.config.audio.source == "file":
                    self.on_progress(audio_chunk_idx, self._audio.total_chunks)
            if self.config.audio.source == "file" and self._chunk_count == 0:
                error_bus.emit("pipeline", "No speech detected in audio file", "warning")
        except Exception as e:
            logger.error("Pipeline error: %s", e)
            error_bus.emit("pipeline", str(e))
        finally:
            self._shutdown()

    def stop(self) -> None:
        self._running = False
        self._worker.stop()
        self._audio.stop()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _process_chunk(self, audio: np.ndarray) -> None:
        if self._paused_event.is_set():
            return

        try:
            text = self._transcriber.transcribe(audio, self.config.audio.sample_rate)
        except Exception as e:
            logger.error("Transcription error: %s", e)
            error_bus.emit("transcriber", str(e))
            return

        if not text:
            return

        self._chunk_count += 1
        self._since_last_summary.append(text)
        self._full_transcript.append(text)
        self.on_transcript(text)

        if self._notes:
            self._notes.append_transcript(text)

        if self._chunk_count % self.config.summarization.summarize_every == 0:
            self._since_last_summary = []
            self._worker.enqueue(" ".join(self._full_transcript))

    def _on_summary_complete(self, summary: str) -> None:
        self._full_summary = summary
        if self._notes:
            self._notes.update_summary(summary)

    def _route_error(self, source: str, message: str, severity: str) -> None:
        self.on_error(source, message, severity)

    def _shutdown(self) -> None:
        self._worker.stop()
        self._worker.join(timeout=5)
        self._transcriber.close()
        error_bus.unsubscribe(self._route_error)

        if self._notes and self._full_summary:
            try:
                loop = asyncio.new_event_loop()
                try:
                    title = loop.run_until_complete(
                        self._summarizer.generate_title(self._full_summary)
                    )
                finally:
                    loop.close()
                if title:
                    new_name = f"{title} — {self._notes._date_str} #Note Assistant"
                    self._notes.set_title(new_name)
            except Exception as e:
                logger.warning("Could not generate note title: %s", e)

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
