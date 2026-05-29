import pytest
from note_assistant.config import (
    AppConfig, AudioConfig, TranscriptionConfig, SummarizationConfig, OutputConfig
)


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
