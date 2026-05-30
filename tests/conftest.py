import pytest
import numpy as np
from note_assistant.config import (
    AppConfig, AudioConfig, TranscriptionConfig, SummarizationConfig, OutputConfig
)
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
    async def summarize(self, transcript: str):
        yield "summary"


@pytest.fixture(autouse=True)
def reset_error_bus():
    """Clear ErrorBus subscribers between tests to prevent cross-test bleed."""
    yield
    try:
        from note_assistant.errors import error_bus
        error_bus._subscribers.clear()
    except ImportError:
        pass


@pytest.fixture
def mock_config(tmp_path):
    return AppConfig(
        audio=AudioConfig(source="mic", chunk_seconds=2.0),
        transcription=TranscriptionConfig(backend="faster-whisper"),
        summarization=SummarizationConfig(backend="ollama", summarize_every=3),
        output=OutputConfig(
            apple_notes=False,
            save_transcript=False,
            save_summary=False,
            output_dir=tmp_path,
        ),
    )


@pytest.fixture
def mock_transcriber():
    return MockTranscriber()


@pytest.fixture
def mock_summarizer():
    return MockSummarizer()


@pytest.fixture
def failing_transcriber():
    return FailingTranscriber()


@pytest.fixture
def mock_app(mock_config, mock_transcriber, mock_summarizer):
    from note_assistant.app import NoteAssistantApp
    return NoteAssistantApp(
        mock_config,
        transcriber=mock_transcriber,
        summarizer=mock_summarizer,
    )
