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


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~3 chars per token (conservative for multilingual text)."""
    return max(1, len(text) // 3)


def _is_repetitive(text: str, fragment_len: int = 30, max_count: int = 4) -> bool:
    """Return True if any substring of fragment_len chars appears more than max_count times."""
    if len(text) < fragment_len * (max_count + 1):
        return False
    seen: dict[str, int] = {}
    for i in range(len(text) - fragment_len):
        frag = text[i : i + fragment_len]
        seen[frag] = seen.get(frag, 0) + 1
        if seen[frag] > max_count:
            return True
    return False


class SummarizationWorker(threading.Thread):
    """Consumes transcript windows from a queue and summarizes them asynchronously."""

    MAX_QUEUE_DEPTH = 3

    def __init__(
        self,
        summarizers: "list[BaseSummarizer]",
        on_summary_token: Callable[[str], None],
        on_summary_complete: Callable[[str], None],
        on_summary_start: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self._summarizers = summarizers
        self._on_summary_token = on_summary_token
        self._on_summary_complete = on_summary_complete
        self._on_summary_start = on_summary_start or (lambda: None)
        self._q: queue.Queue[str | None] = queue.Queue()
        self._paused_event = threading.Event()
        self._last_queue_warning = 0.0
        self._warmed: set[int] = set()

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
        est_tokens = _estimate_tokens(window)
        for idx, summarizer in enumerate(self._summarizers):
            limit = summarizer.TOKEN_LIMIT
            if limit:
                pct = est_tokens / limit
                if pct >= 0.70 and idx + 1 < len(self._summarizers) and (idx + 1) not in self._warmed:
                    # Background warmup of next fallback backend at 70%
                    next_s = self._summarizers[idx + 1]
                    self._warmed.add(idx + 1)
                    t = threading.Thread(target=next_s.warmup, daemon=True, name=f"warmup-{idx+1}")
                    t.start()
                    logger.info("Triggered background warmup of backend %d at ~%.0f%% token usage", idx + 1, pct * 100)
                if pct >= 0.90:
                    logger.info("Skipping backend %d: ~%d tokens = %.0f%% of %d-token limit", idx, est_tokens, pct * 100, limit)
                    continue
            last = ""
            try:
                async for chunk in summarizer.summarize(window):
                    if not last:
                        self._on_summary_start()
                    last = chunk
                    self._on_summary_token(chunk)
            except Exception as e:
                logger.error("Summarization error (backend %d): %s", idx, e)
                error_bus.emit("summarizer", str(e))
                last = ""

            if last and not _is_repetitive(last):
                self._on_summary_complete(last)
                return
            if last:
                logger.warning("Repetitive summary from backend %d, trying fallback", idx)

        if last:  # all backends repetitive — use last result anyway
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

        summarizers = [self._summarizer]
        if summarizer is None:
            for backend in ("mlx", "ollama", "apple"):
                if backend == config.summarization.backend:
                    continue
                try:
                    fb_cfg = config.summarization.model_copy(update={"backend": backend})
                    summarizers.append(create_summarizer(
                        fb_cfg,
                        language_input=config.language_input,
                        language_output=config.language_output,
                    ))
                    logger.debug("Registered fallback summarizer: %s", backend)
                    break
                except Exception:
                    pass

        self._worker = SummarizationWorker(
            summarizers,
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
            title = ""
            est = _estimate_tokens(self._full_summary)
            # Pick first summarizer whose token limit can fit the summary
            title_summarizer = next(
                (s for s in self._worker._summarizers
                 if s.TOKEN_LIMIT is None or est <= int(s.TOKEN_LIMIT * 0.90)),
                self._worker._summarizers[-1],
            )
            try:
                loop = asyncio.new_event_loop()
                try:
                    title = loop.run_until_complete(
                        title_summarizer.generate_title(self._full_summary)
                    )
                finally:
                    loop.close()
            except Exception as e:
                logger.warning("Could not generate note title via LLM: %s", e)
            if not title:
                # Fallback: first bullet of summary, up to 6 words
                first = self._full_summary.split("\n")[0].strip().lstrip("- ").lstrip("* ")
                title = " ".join(first.split()[:6])
            if title:
                self._notes.set_title(f"{title} — {self._notes._date_str} #Note Assistant")

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
