import platform
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
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


async def test_ollama_summarizer_streams_tokens():
    with patch.object(OllamaSummarizer, "_load"):
        cfg = SummarizationConfig(backend="ollama", ollama_model="llama3.2:3b")
        s = OllamaSummarizer(cfg, "English", "English")

        async def fake_stream():
            for content in ["bullet", " one"]:
                yield {"message": {"content": content}}

        s._ollama = MagicMock()
        s._ollama.chat = AsyncMock(return_value=fake_stream())
        result = "".join([t async for t in s.summarize("some transcript")])
        assert result == "bullet one"


async def test_ollama_summarizer_translation_appends_instruction():
    with patch.object(OllamaSummarizer, "_load"):
        cfg = SummarizationConfig(backend="ollama")
        s = OllamaSummarizer(cfg, "English", "Thai")
        prompts = []

        async def fake_stream():
            return
            yield  # empty async generator

        async def fake_chat(**kwargs):
            prompts.append(kwargs["messages"][0]["content"])
            return fake_stream()

        s._ollama = MagicMock()
        s._ollama.chat = fake_chat
        async for _ in s.summarize("hello"):
            pass
        assert "Thai" in prompts[0]
