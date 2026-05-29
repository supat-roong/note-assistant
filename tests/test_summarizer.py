import platform
import pytest
from unittest.mock import patch
from note_assistant.summarizer import OllamaSummarizer, create_summarizer, _REGISTRY
from note_assistant.config import SummarizationConfig


def test_registry_contains_expected_backends():
    assert "ollama" in _REGISTRY
    assert "mlx" in _REGISTRY
    if platform.system() == "Darwin":
        assert "apple" in _REGISTRY


def test_create_summarizer_unknown_backend_raises():
    cfg = SummarizationConfig()
    cfg.__dict__["backend"] = "unknown"
    with pytest.raises(ValueError, match="Unknown summarization backend: unknown"):
        create_summarizer(cfg)


def test_create_summarizer_does_not_mutate_config():
    """Fallback path must not mutate the caller's config object."""
    cfg = SummarizationConfig(backend="apple")
    original_backend = cfg.backend
    try:
        create_summarizer(cfg)
    except Exception:
        pass
    assert cfg.backend == original_backend


def test_ollama_summarizer_streams_tokens():
    with patch.object(OllamaSummarizer, "_load"):
        cfg = SummarizationConfig(backend="ollama", ollama_model="llama3.2:3b")
        s = OllamaSummarizer(cfg, "English", "English")
        s._ollama = type("FakeOllama", (), {
            "chat": lambda self, **kw: [
                {"message": {"content": "bullet"}},
                {"message": {"content": " one"}},
            ]
        })()
        result = "".join(s.summarize("some transcript"))
        assert result == "bullet one"


def test_ollama_summarizer_translation_appends_instruction():
    with patch.object(OllamaSummarizer, "_load"):
        cfg = SummarizationConfig(backend="ollama")
        s = OllamaSummarizer(cfg, "English", "Thai")
        prompts = []
        def fake_chat(model, messages, stream, options):
            prompts.append(messages[0]["content"])
            return []
        s._ollama = type("FO", (), {"chat": staticmethod(fake_chat)})()
        list(s.summarize("hello"))
        assert "Thai" in prompts[0]
