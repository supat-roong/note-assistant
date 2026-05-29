from pathlib import Path
from note_assistant.config import AppConfig, load_config, detect_best_backends


def test_default_log_level():
    cfg = AppConfig()
    assert cfg.log_level == "WARNING"


def test_load_config_defaults():
    cfg = load_config(path=None)
    assert cfg.audio.source == "mic"
    assert cfg.transcription.backend == "apple"
    assert cfg.log_level == "WARNING"


def test_load_config_from_yaml(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("audio:\n  source: system\n  chunk_seconds: 3.0\nlog_level: DEBUG\n")
    cfg = load_config(path=p)
    assert cfg.audio.source == "system"
    assert cfg.audio.chunk_seconds == 3.0
    assert cfg.log_level == "DEBUG"


def test_load_config_override():
    cfg = load_config(path=None, source="system")
    assert cfg.audio.source == "system"


def test_detect_best_backends_returns_valid_tuple():
    t, s = detect_best_backends()
    assert t in ("apple", "faster-whisper")
    assert s in ("apple", "mlx", "ollama")
