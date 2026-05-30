import importlib.util
import platform
import numpy as np
import pytest
from unittest.mock import patch, Mock
from note_assistant.transcriber import (
    FasterWhisperTranscriber, MLXWhisperTranscriber, create_transcriber, _REGISTRY
)
from note_assistant.config import TranscriptionConfig

_mlx_available = importlib.util.find_spec("mlx_whisper") is not None


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
        t._ready.set()  # simulate successful background load
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
        t._ready.set()  # simulate successful background load
        t._model = Mock()
        t._model.transcribe.return_value = ([], {})
        audio = np.zeros(1600, dtype=np.float32)
        assert t.transcribe(audio, 16000) == ""


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
def test_create_apple_transcriber_on_darwin():
    cfg = TranscriptionConfig(backend="apple")
    from note_assistant.transcriber import AppleSpeechTranscriber
    assert _REGISTRY["apple"] is AppleSpeechTranscriber


def test_transcription_config_has_mlx_whisper_model_field():
    cfg = TranscriptionConfig(backend="mlx-whisper", mlx_whisper_model="mlx-community/whisper-base-mlx")
    assert cfg.mlx_whisper_model == "mlx-community/whisper-base-mlx"
    assert cfg.backend == "mlx-whisper"


def test_transcription_config_mlx_whisper_model_default():
    cfg = TranscriptionConfig()
    assert cfg.mlx_whisper_model == "mlx-community/whisper-base-mlx"


@pytest.mark.skipif(platform.machine() != "arm64", reason="arm64 only")
def test_registry_contains_mlx_whisper_on_arm64():
    assert "mlx-whisper" in _REGISTRY


@pytest.mark.skipif(not _mlx_available, reason="mlx-whisper not installed")
def test_mlx_whisper_transcribes_text():
    with patch.object(MLXWhisperTranscriber, "_load"):
        cfg = TranscriptionConfig(backend="mlx-whisper", mlx_whisper_model="mlx-community/whisper-base-mlx")
        t = MLXWhisperTranscriber(cfg)
        t._ready.set()
        mock_mlx = Mock()
        mock_mlx.transcribe.return_value = {"text": " Hello world"}
        t._mlx_whisper = mock_mlx
        audio = np.ones(1600, dtype=np.float32)
        assert t.transcribe(audio, 16000) == "Hello world"


@pytest.mark.skipif(not _mlx_available, reason="mlx-whisper not installed")
def test_mlx_whisper_skips_while_loading():
    with patch.object(MLXWhisperTranscriber, "_load"):
        cfg = TranscriptionConfig(backend="mlx-whisper")
        t = MLXWhisperTranscriber(cfg)
        # _ready not set — model still loading
        audio = np.ones(1600, dtype=np.float32)
        assert t.transcribe(audio, 16000) == ""


@pytest.mark.skipif(not _mlx_available, reason="mlx-whisper not installed")
def test_mlx_whisper_surfaces_load_error_once():
    with patch.object(MLXWhisperTranscriber, "_load"):
        cfg = TranscriptionConfig(backend="mlx-whisper")
        t = MLXWhisperTranscriber(cfg)
        t._load_error = "download failed"
        t._ready.set()
        audio = np.ones(1600, dtype=np.float32)
        with pytest.raises(RuntimeError, match="MLX Whisper model failed to load"):
            t.transcribe(audio, 16000)
        # second call is silent
        assert t.transcribe(audio, 16000) == ""


@pytest.mark.skipif(not _mlx_available, reason="mlx-whisper not installed")
def test_mlx_whisper_passes_language_to_transcribe():
    with patch.object(MLXWhisperTranscriber, "_load"):
        cfg = TranscriptionConfig(backend="mlx-whisper", language="th")
        t = MLXWhisperTranscriber(cfg)
        t._ready.set()
        mock_mlx = Mock()
        mock_mlx.transcribe.return_value = {"text": "สวัสดี"}
        t._mlx_whisper = mock_mlx
        audio = np.ones(1600, dtype=np.float32)
        t.transcribe(audio, 16000)
        _, kwargs = mock_mlx.transcribe.call_args
        assert kwargs.get("language") == "th"


@pytest.mark.skipif(not _mlx_available, reason="mlx-whisper not installed")
def test_mlx_whisper_auto_language_omits_language_kwarg():
    with patch.object(MLXWhisperTranscriber, "_load"):
        cfg = TranscriptionConfig(backend="mlx-whisper", language=None)
        t = MLXWhisperTranscriber(cfg)
        t._ready.set()
        mock_mlx = Mock()
        mock_mlx.transcribe.return_value = {"text": "hello"}
        t._mlx_whisper = mock_mlx
        audio = np.ones(1600, dtype=np.float32)
        t.transcribe(audio, 16000)
        _, kwargs = mock_mlx.transcribe.call_args
        assert "language" not in kwargs
