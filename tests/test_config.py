from pathlib import Path
from note_assistant.config import AppConfig, OutputConfig, load_config, detect_best_backends


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


def test_load_config_log_level_override():
    cfg = load_config(path=None, log_level="DEBUG")
    assert cfg.log_level == "DEBUG"


def test_output_config_auto_title_default():
    assert OutputConfig().auto_title is True


def test_output_config_title_prompt_template_has_placeholders():
    rendered = OutputConfig().title_prompt_template.format(language="English", summary="notes")
    assert "English" in rendered
    assert "notes" in rendered


def test_load_config_auto_title_false_from_yaml(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("output:\n  auto_title: false\n")
    cfg = load_config(path=p)
    assert cfg.output.auto_title is False


def test_load_config_custom_title_prompt_from_yaml(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text('output:\n  title_prompt_template: "Custom {language} {summary}"\n')
    cfg = load_config(path=p)
    assert cfg.output.title_prompt_template == "Custom {language} {summary}"
