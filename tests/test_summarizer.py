import platform
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from note_assistant.summarizer import (
    AppleFoundationSummarizer, OllamaSummarizer, create_summarizer, _REGISTRY
)
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
        s._ollama.chat = AsyncMock(side_effect=lambda **kw: fake_stream())
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


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
async def test_apple_summarizer_raises_when_sdk_missing():
    import sys
    with patch.dict(sys.modules, {"apple_fm_sdk": None}):
        with pytest.raises(RuntimeError, match="apple-fm-sdk not installed"):
            AppleFoundationSummarizer(SummarizationConfig())


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
async def test_apple_summarizer_raises_when_model_unavailable():
    mock_fm = MagicMock()
    mock_fm.SystemLanguageModel.return_value.is_available.return_value = (False, "not enabled")
    import sys
    with patch.dict(sys.modules, {"apple_fm_sdk": mock_fm}):
        with pytest.raises(RuntimeError, match="not available"):
            AppleFoundationSummarizer(SummarizationConfig())


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
async def test_apple_summarizer_streams_tokens():
    mock_fm = MagicMock()
    mock_fm.SystemLanguageModel.return_value.is_available.return_value = (True, None)

    async def fake_stream():
        yield "bullet"
        yield " one"

    mock_session = MagicMock()
    mock_session.stream_response = MagicMock(return_value=fake_stream())
    mock_fm.LanguageModelSession.return_value = mock_session

    import sys
    with patch.dict(sys.modules, {"apple_fm_sdk": mock_fm}):
        s = AppleFoundationSummarizer(SummarizationConfig())
        result = "".join([t async for t in s.summarize("meeting notes")])
        assert result == "bullet one"


async def test_ollama_generate_title_uses_template():
    with patch.object(OllamaSummarizer, "_load"):
        cfg = SummarizationConfig(backend="ollama", ollama_model="llama3.2:3b")
        s = OllamaSummarizer(cfg, "English", "English")
        s._ollama = MagicMock()
        s._ollama.chat = AsyncMock(return_value={"message": {"content": "My Title"}})
        template = "Title in {language}:\n\n{summary}"
        result = await s.generate_title("- bullet one\n- bullet two", template)
        assert result == "My Title"
        sent = s._ollama.chat.call_args.kwargs["messages"][0]["content"]
        assert "English" in sent
        assert "bullet one" in sent


async def test_ollama_generate_title_strips_punctuation():
    with patch.object(OllamaSummarizer, "_load"):
        cfg = SummarizationConfig(backend="ollama")
        s = OllamaSummarizer(cfg, "English", "English")
        s._ollama = MagicMock()
        s._ollama.chat = AsyncMock(return_value={"message": {"content": '"Title."'}})
        result = await s.generate_title("notes", "t {language} {summary}")
        assert result == "Title"


def test_base_summarizer_generate_title_returns_empty():
    from note_assistant.summarizer import BaseSummarizer
    import asyncio

    class Minimal(BaseSummarizer):
        async def summarize(self, transcript):
            yield ""

    s = Minimal()
    result = asyncio.run(s.generate_title("summary text", "prompt {language} {summary}"))
    assert result == ""
