import asyncio
import time
import numpy as np
import pytest
from note_assistant.summarizer import BaseSummarizer


class SlowSummarizer(BaseSummarizer):
    async def summarize(self, transcript: str):
        await asyncio.sleep(0.2)
        yield "done"


# ---------------------------------------------------------------------------
# SummarizationWorker tests
# ---------------------------------------------------------------------------

from note_assistant.app import SummarizationWorker


def test_worker_enqueue_is_non_blocking():
    worker = SummarizationWorker(
        [SlowSummarizer()],
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
        [MockSummarizer()],
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
        [MockSummarizer()],
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
        [MockSummarizer()],
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
        [MockSummarizer()],
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


def test_shutdown_skips_title_when_auto_title_disabled(mock_config, mock_transcriber, mock_summarizer):
    from unittest.mock import MagicMock
    mock_config.output.auto_title = False
    app = NoteAssistantApp(mock_config, transcriber=mock_transcriber, summarizer=mock_summarizer)
    app._full_summary = "- some meeting notes"
    mock_notes = MagicMock()
    app._notes = mock_notes
    # Replace worker so no threads run
    app._worker = MagicMock()
    app._worker.join = MagicMock()
    app._shutdown()
    mock_notes.set_title.assert_not_called()
    mock_notes.close_session.assert_called_once()


def test_worker_clear_queue_drains_pending_items(mock_summarizer):
    worker = SummarizationWorker(
        [mock_summarizer],
        on_summary_token=lambda t: None,
        on_summary_complete=lambda s: None,
    )
    worker.enqueue("a")
    worker.enqueue("b")
    worker.clear_queue()
    assert worker._q.empty()


def test_shutdown_clears_queue_then_enqueues_final_summary(mock_config, mock_transcriber, mock_summarizer):
    from unittest.mock import MagicMock, call
    app = NoteAssistantApp(mock_config, transcriber=mock_transcriber, summarizer=mock_summarizer)
    app._full_transcript = ["a", "b"]
    app._since_last_summary = ["a", "b"]
    app._worker = MagicMock()
    app._shutdown()
    calls = app._worker.method_calls
    assert call.clear_queue() in calls
    assert call.enqueue("a b") in calls
    assert calls.index(call.clear_queue()) < calls.index(call.enqueue("a b"))


def test_shutdown_enqueues_final_summary_when_unsummarized_chunks_remain(mock_config, mock_transcriber, mock_summarizer):
    from unittest.mock import MagicMock
    app = NoteAssistantApp(mock_config, transcriber=mock_transcriber, summarizer=mock_summarizer)
    app._full_transcript = ["chunk one", "chunk two"]
    app._since_last_summary = ["chunk one", "chunk two"]
    app._worker = MagicMock()
    app._shutdown()
    app._worker.enqueue.assert_called_once_with("chunk one chunk two")


def test_shutdown_skips_final_enqueue_when_no_unsummarized_chunks(mock_config, mock_transcriber, mock_summarizer):
    from unittest.mock import MagicMock
    app = NoteAssistantApp(mock_config, transcriber=mock_transcriber, summarizer=mock_summarizer)
    app._full_transcript = ["chunk one", "chunk two"]
    app._since_last_summary = []
    app._worker = MagicMock()
    app._shutdown()
    app._worker.enqueue.assert_not_called()


def test_launch_closes_terminal_on_return_code_99():
    import os
    import signal
    from unittest.mock import MagicMock, patch
    from note_assistant.__main__ import _launch
    from note_assistant.config import AppConfig, AudioConfig, TranscriptionConfig, SummarizationConfig, OutputConfig

    cfg = AppConfig(
        audio=AudioConfig(source="mic"),
        transcription=TranscriptionConfig(backend="faster-whisper"),
        summarization=SummarizationConfig(backend="ollama"),
        output=OutputConfig(apple_notes=False, save_transcript=False, save_summary=False),
    )

    mock_ui = MagicMock()
    mock_ui.return_code = 99

    popen_calls = []
    kill_calls = []

    with patch("note_assistant.ui.NoteAssistantUI", return_value=mock_ui), \
         patch("note_assistant.app.NoteAssistantApp"), \
         patch("subprocess.Popen", side_effect=lambda *a, **kw: popen_calls.append(a[0])), \
         patch("os.kill", side_effect=lambda pid, sig: kill_calls.append((pid, sig))), \
         patch("os.getppid", return_value=12345), \
         patch("sys.exit"):
        _launch(cfg)

    assert any("osascript" in str(c) for c in popen_calls), "osascript not called"
    assert any("close front window" in str(c) for c in popen_calls)
    assert (12345, signal.SIGKILL) in kill_calls


def test_shutdown_calls_close_on_all_summarizers():
    """_shutdown() must call close() on every summarizer in the worker."""
    from unittest.mock import MagicMock
    from note_assistant.app import NoteAssistantApp
    from note_assistant.config import AppConfig, AudioConfig, TranscriptionConfig, SummarizationConfig, OutputConfig
    from note_assistant.transcriber import BaseTranscriber
    from conftest import MockSummarizer
    import numpy as np

    class TrackingMockSummarizer(MockSummarizer):
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class NullTranscriber(BaseTranscriber):
        def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
            return ""

    s1 = TrackingMockSummarizer()
    s2 = TrackingMockSummarizer()

    cfg = AppConfig(
        audio=AudioConfig(source="mic"),
        transcription=TranscriptionConfig(backend="faster-whisper"),
        summarization=SummarizationConfig(backend="ollama"),
        output=OutputConfig(apple_notes=False, save_transcript=False, save_summary=False),
    )

    app = NoteAssistantApp(cfg, transcriber=NullTranscriber(), summarizer=s1)
    # Inject second summarizer directly into the worker's list
    app._worker._summarizers.append(s2)

    app._worker.start()
    app._shutdown()

    assert s1.closed, "Primary summarizer was not closed"
    assert s2.closed, "Fallback summarizer was not closed"


def test_launch_does_not_close_terminal_on_normal_exit():
    from unittest.mock import MagicMock, patch
    from note_assistant.__main__ import _launch
    from note_assistant.config import AppConfig, AudioConfig, TranscriptionConfig, SummarizationConfig, OutputConfig

    cfg = AppConfig(
        audio=AudioConfig(source="mic"),
        transcription=TranscriptionConfig(backend="faster-whisper"),
        summarization=SummarizationConfig(backend="ollama"),
        output=OutputConfig(apple_notes=False, save_transcript=False, save_summary=False),
    )

    mock_ui = MagicMock()
    mock_ui.return_code = 0

    popen_calls = []

    with patch("note_assistant.ui.NoteAssistantUI", return_value=mock_ui), \
         patch("note_assistant.app.NoteAssistantApp"), \
         patch("subprocess.Popen", side_effect=lambda *a, **kw: popen_calls.append(a[0])):
        _launch(cfg)

    assert not popen_calls
