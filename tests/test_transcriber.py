import platform
import numpy as np
import pytest
from unittest.mock import patch, Mock
from note_assistant.transcriber import (
    FasterWhisperTranscriber, create_transcriber, _REGISTRY
)
from note_assistant.config import TranscriptionConfig


def test_registry_contains_expected_backends():
    assert "faster-whisper" in _REGISTRY
    if platform.system() == "Darwin":
        assert "apple" in _REGISTRY


def test_create_transcriber_unknown_backend_raises():
    cfg = TranscriptionConfig()
    cfg.__dict__["backend"] = "unknown"
    with pytest.raises(ValueError, match="Unknown transcription backend: unknown"):
        create_transcriber(cfg)


def test_faster_whisper_transcribes_text():
    with patch.object(FasterWhisperTranscriber, "_load"):
        cfg = TranscriptionConfig(backend="faster-whisper", whisper_model="tiny")
        t = FasterWhisperTranscriber(cfg)
        seg = Mock()
        seg.text = " Hello world"
        t._model = Mock()
        t._model.transcribe.return_value = ([seg], {})
        audio = np.ones(1600, dtype=np.float32)
        assert t.transcribe(audio, 16000) == "Hello world"


def test_faster_whisper_empty_segments_returns_empty():
    with patch.object(FasterWhisperTranscriber, "_load"):
        cfg = TranscriptionConfig(backend="faster-whisper", whisper_model="tiny")
        t = FasterWhisperTranscriber(cfg)
        t._model = Mock()
        t._model.transcribe.return_value = ([], {})
        audio = np.zeros(1600, dtype=np.float32)
        assert t.transcribe(audio, 16000) == ""


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
def test_create_apple_transcriber_on_darwin():
    cfg = TranscriptionConfig(backend="apple")
    from note_assistant.transcriber import AppleSpeechTranscriber
    assert _REGISTRY["apple"] is AppleSpeechTranscriber
