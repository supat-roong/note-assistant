import time
import numpy as np
import pytest
from note_assistant.summarizer import BaseSummarizer


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
    from conftest import MockSummarizer
    worker = SummarizationWorker(
        MockSummarizer(),
        on_summary_token=lambda t: None,
        on_summary_complete=results.append,
    )
    worker.start()
    worker.enqueue("some text")
    worker.stop()
    worker.join(timeout=2)
    assert results == ["summary"]


def test_worker_stops_via_sentinel():
    from conftest import MockSummarizer
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
    from conftest import MockSummarizer
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
    from conftest import MockSummarizer
    processed = []
    worker = SummarizationWorker(
        MockSummarizer(),
        on_summary_token=lambda t: None,
        on_summary_complete=processed.append,
    )
    worker.start()
    worker.pause()
    worker.enqueue("should be discarded")
    worker.stop()
    worker.join(timeout=2)
    assert processed == []


# ---------------------------------------------------------------------------
# NoteAssistantApp tests
# ---------------------------------------------------------------------------

from note_assistant.app import NoteAssistantApp
from note_assistant.errors import error_bus


def test_process_chunk_increments_count(mock_app):
    chunk = np.ones(1600, dtype=np.float32)
    mock_app._process_chunk(chunk)
    assert mock_app.chunk_count == 1


def test_process_chunk_calls_on_transcript(mock_app):
    received = []
    mock_app.on_transcript = received.append
    chunk = np.ones(1600, dtype=np.float32)
    mock_app._process_chunk(chunk)
    assert received == ["hello world"]


def test_process_chunk_skips_when_paused(mock_app):
    mock_app.pause()
    chunk = np.ones(1600, dtype=np.float32)
    mock_app._process_chunk(chunk)
    assert mock_app.chunk_count == 0


def test_pause_and_resume(mock_app):
    mock_app.pause()
    assert mock_app._paused_event.is_set()
    mock_app.resume()
    assert not mock_app._paused_event.is_set()


def test_error_recovery_continues_pipeline(mock_config, failing_transcriber, mock_summarizer):
    errors = []
    error_bus.subscribe(lambda s, m, sev: errors.append(m))
    app = NoteAssistantApp(
        mock_config,
        transcriber=failing_transcriber,
        summarizer=mock_summarizer,
    )
    chunk = np.zeros(1600, dtype=np.float32)
    for _ in range(5):
        app._process_chunk(chunk)
    assert len(errors) == 1
    assert app.chunk_count == 4  # chunks 1,2,4,5 succeeded; 3 failed


def test_on_error_callback_fires(mock_config, failing_transcriber, mock_summarizer):
    received = []
    app = NoteAssistantApp(
        mock_config,
        transcriber=failing_transcriber,
        summarizer=mock_summarizer,
        on_error=lambda s, m, sev: received.append((s, m)),
    )
    chunk = np.zeros(1600, dtype=np.float32)
    for _ in range(3):
        app._process_chunk(chunk)
    assert len(received) == 1
    assert received[0][0] == "transcriber"
