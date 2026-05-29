import time
import numpy as np
import pytest
from note_assistant.transcriber import BaseTranscriber
from note_assistant.summarizer import BaseSummarizer


class MockTranscriber(BaseTranscriber):
    def __init__(self, text: str = "hello world"):
        self._text = text
        self.call_count = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        self.call_count += 1
        return self._text


class FailingTranscriber(BaseTranscriber):
    def __init__(self):
        self.call_count = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        self.call_count += 1
        if self.call_count == 3:
            raise RuntimeError("Simulated transcription failure")
        return "test text"


class MockSummarizer(BaseSummarizer):
    def __init__(self, tokens=None):
        self._tokens = tokens or ["summary"]

    def summarize(self, transcript: str):
        yield from self._tokens


class SlowSummarizer(BaseSummarizer):
    def summarize(self, transcript: str):
        time.sleep(0.2)
        yield "done"


# ---------------------------------------------------------------------------
# SummarizationWorker tests
# ---------------------------------------------------------------------------

from note_assistant.app import SummarizationWorker


def test_worker_enqueue_is_non_blocking():
    worker = SummarizationWorker(
        SlowSummarizer(),
        on_summary_token=lambda t: None,
        on_summary_complete=lambda s: None,
    )
    worker.start()
    start = time.monotonic()
    worker.enqueue("window 1")
    worker.enqueue("window 2")
    elapsed = time.monotonic() - start
    worker.stop()
    worker.join(timeout=2)
    assert elapsed < 0.05  # enqueue never blocks


def test_worker_processes_enqueued_window():
    results = []
    worker = SummarizationWorker(
        MockSummarizer(["result"]),
        on_summary_token=lambda t: None,
        on_summary_complete=results.append,
    )
    worker.start()
    worker.enqueue("some text")
    worker.stop()
    worker.join(timeout=2)
    assert results == ["result"]


def test_worker_stops_via_sentinel():
    worker = SummarizationWorker(
        MockSummarizer(),
        on_summary_token=lambda t: None,
        on_summary_complete=lambda s: None,
    )
    worker.start()
    worker.stop()
    worker.join(timeout=2)
    assert not worker.is_alive()


def test_worker_drops_oldest_when_queue_full():
    from note_assistant.errors import error_bus
    warnings = []
    error_bus.subscribe(lambda s, m, sev: warnings.append(sev) if sev == "warning" else None)

    worker = SummarizationWorker(
        MockSummarizer(),
        on_summary_token=lambda t: None,
        on_summary_complete=lambda s: None,
    )
    # Do NOT start worker — items stay in queue so we can measure depth
    for i in range(SummarizationWorker.MAX_QUEUE_DEPTH + 1):
        worker.enqueue(f"window {i}")

    assert worker._q.qsize() == SummarizationWorker.MAX_QUEUE_DEPTH
    assert len(warnings) == 1


def test_worker_pauses_and_discards_windows():
    processed = []
    worker = SummarizationWorker(
        MockSummarizer(["token"]),
        on_summary_token=lambda t: None,
        on_summary_complete=processed.append,
    )
    worker.start()
    worker.pause()
    worker.enqueue("should be discarded")
    worker.stop()
    worker.join(timeout=2)
    assert processed == []
