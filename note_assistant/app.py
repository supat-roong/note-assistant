"""Main orchestrator — ties audio, transcription, summarization, notes together."""
from __future__ import annotations

import asyncio
import math
import queue
import threading
import time
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


_EWMA_ALPHA = 0.25  # weight for most-recent sample in exponential moving average


class SummarizationWorker(threading.Thread):
    """Consumes transcript windows from a queue and summarizes them asynchronously."""

    MAX_QUEUE_DEPTH = 3

    def __init__(
        self,
        summarizers: "list[BaseSummarizer]",
        on_summary_token: Callable[[str], None],
        on_summary_complete: Callable[[str], None],
        on_summary_start: Callable[[], None] | None = None,
        on_backend_switch: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self._summarizers = summarizers
        self._on_summary_token = on_summary_token
        self._on_summary_complete = on_summary_complete
        self._on_summary_start = on_summary_start or (lambda: None)
        self._on_backend_switch = on_backend_switch or (lambda label: None)
        self._q: queue.Queue[str | None] = queue.Queue()
        self._paused_event = threading.Event()
        self._last_queue_warning = 0.0
        self._warmed: set[int] = set()
        self._avg_sum_seconds: float | None = None
        self._consec_failures: list[int] = [0] * len(summarizers)
        self._last_active_idx: int = 0

    @property
    def avg_summarization_seconds(self) -> float | None:
        return self._avg_sum_seconds

    def enqueue(self, window: str) -> bool:
        """Enqueue a transcript window. Returns False if the queue was full and a window was dropped."""
        dropped = False
        if self._q.qsize() >= self.MAX_QUEUE_DEPTH:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            dropped = True
            logger.warning("summarization queue full — dropping window")
            now = time.monotonic()
            if now - self._last_queue_warning > 10.0:
                self._last_queue_warning = now
                error_bus.emit("summarizer", "Queue full — dropping oldest window", "warning")
        self._q.put(window)
        return not dropped

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
        _t0 = time.monotonic()
        est_tokens = _estimate_tokens(window)
        for idx, summarizer in enumerate(self._summarizers):
            if self._consec_failures[idx] >= 2:
                logger.info("Skipping backend %d: %d consecutive failures", idx, self._consec_failures[idx])
                continue
            if idx > 0 and idx != self._last_active_idx:
                self._on_backend_switch(summarizer.model_label)
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
                    last += chunk
                    self._on_summary_token(chunk)
            except Exception as e:
                self._consec_failures[idx] += 1
                has_fallback = idx + 1 < len(self._summarizers)
                logger.error("Summarization error (backend %d): %s", idx, e)
                error_bus.emit("summarizer", str(e), "warning" if has_fallback else "error")
                last = ""

            if last and not _is_repetitive(last):
                self._consec_failures[idx] = 0
                self._last_active_idx = idx
                elapsed = time.monotonic() - _t0
                if self._avg_sum_seconds is None:
                    self._avg_sum_seconds = elapsed
                else:
                    self._avg_sum_seconds = _EWMA_ALPHA * elapsed + (1 - _EWMA_ALPHA) * self._avg_sum_seconds
                logger.debug("Summarization took %.1fs (EWMA %.1fs)", elapsed, self._avg_sum_seconds)
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
        on_backend_switch: Callable[[str], None] | None = None,
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
        self.on_backend_switch = on_backend_switch or (lambda label: None)

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
        self._effective_summarize_every = config.summarization.summarize_every
        self._last_chunk_time: float | None = None
        self._avg_chunk_seconds: float | None = None

        summarizers = [self._summarizer]
        if summarizer is None:
            kw = dict(language_input=config.language_input, language_output=config.language_output)
            sc = config.summarization
            candidates: list[tuple[str, dict]] = []
            if sc.backend != "mlx":
                candidates.append(("mlx", {"mlx_model": sc.mlx_model}))
                candidates.append(("mlx", {"mlx_model": sc.mlx_fallback_model}))
            else:
                candidates.append(("mlx", {"mlx_model": sc.mlx_fallback_model}))
            if sc.backend != "ollama":
                candidates.append(("ollama", {"ollama_model": sc.ollama_model}))
                candidates.append(("ollama", {"ollama_model": sc.ollama_fallback_model}))
            else:
                candidates.append(("ollama", {"ollama_model": sc.ollama_fallback_model}))

            for backend, overrides in candidates:
                try:
                    fb_cfg = sc.model_copy(update={"backend": backend, **overrides})
                    fb = create_summarizer(fb_cfg, **kw)
                    summarizers.append(fb)
                    logger.debug("Registered fallback: %s %s", backend, overrides)
                except Exception as e:
                    logger.debug("Could not register fallback %s %s: %s", backend, overrides, e)

        self._worker = SummarizationWorker(
            summarizers,
            on_summary_token=self.on_summary,
            on_summary_complete=self._on_summary_complete,
            on_summary_start=self.on_summary_start,
            on_backend_switch=self.on_backend_switch,
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

        # Track chunk arrival rate for adaptive interval estimation
        now = time.monotonic()
        if self._last_chunk_time is not None:
            interval = now - self._last_chunk_time
            if self._avg_chunk_seconds is None:
                self._avg_chunk_seconds = interval
            else:
                self._avg_chunk_seconds = _EWMA_ALPHA * interval + (1 - _EWMA_ALPHA) * self._avg_chunk_seconds
        self._last_chunk_time = now

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

        if self._chunk_count % self._effective_summarize_every == 0:
            self._since_last_summary = []
            queued = self._worker.enqueue(" ".join(self._full_transcript))
            base = self.config.summarization.summarize_every
            avg_sum = self._worker.avg_summarization_seconds
            avg_chunk = self._avg_chunk_seconds or self.config.audio.chunk_seconds
            if avg_sum is not None:
                n_needed = math.ceil(avg_sum / avg_chunk)
                self._effective_summarize_every = max(base, min(n_needed, base * 8))
                logger.info(
                    "Adaptive interval: sum=%.1fs chunk=%.1fs → every %d chunks",
                    avg_sum, avg_chunk, self._effective_summarize_every,
                )
            else:
                # No timing data yet — fall back to queue-depth heuristic
                if not queued:
                    self._effective_summarize_every = min(self._effective_summarize_every * 2, base * 8)
                    logger.info("Queue full — backed off to %d chunks (no timing data yet)", self._effective_summarize_every)
                elif self._effective_summarize_every > base:
                    self._effective_summarize_every = max(base, self._effective_summarize_every - 1)

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

        if self.config.output.auto_title and self._notes and self._full_summary:
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
                        title_summarizer.generate_title(
                            self._full_summary,
                            self.config.output.title_prompt_template,
                        )
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
