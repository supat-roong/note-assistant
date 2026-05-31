import platform
import subprocess
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
    with patch.object(OllamaSummarizer, "_load"), \
         patch.object(OllamaSummarizer, "_ensure_ollama_running"):
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
    with patch.object(OllamaSummarizer, "_load"), \
         patch.object(OllamaSummarizer, "_ensure_ollama_running"):
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
    with patch.object(OllamaSummarizer, "_load"), \
         patch.object(OllamaSummarizer, "_ensure_ollama_running"):
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
    with patch.object(OllamaSummarizer, "_load"), \
         patch.object(OllamaSummarizer, "_ensure_ollama_running"):
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


def test_base_summarizer_close_is_noop():
    from note_assistant.summarizer import BaseSummarizer

    class Minimal(BaseSummarizer):
        async def summarize(self, transcript):
            yield ""

    s = Minimal()
    s.close()  # must not raise


def test_ensure_ollama_running_noop_when_already_running():
    """When Ollama answers /api/tags, _owned_process stays None."""
    with patch("urllib.request.urlopen", return_value=MagicMock()), \
         patch.object(OllamaSummarizer, "_load"), \
         patch("note_assistant.summarizer._ollama_context_length", return_value=4096):
        cfg = SummarizationConfig(backend="ollama")
        s = OllamaSummarizer(cfg)
    assert s._owned_process is None


def test_ensure_ollama_running_starts_server_when_not_running():
    """When /api/tags is unreachable, Ollama is started and _owned_process is set."""
    mock_proc = MagicMock()
    # First call (liveness check) fails, second call (poll after start) succeeds
    side_effects = [OSError("connection refused"), MagicMock()]

    with patch("urllib.request.urlopen", side_effect=side_effects), \
         patch("note_assistant.summarizer.subprocess.Popen", return_value=mock_proc) as mock_popen, \
         patch.object(OllamaSummarizer, "_load"), \
         patch("note_assistant.summarizer._ollama_context_length", return_value=4096):
        cfg = SummarizationConfig(backend="ollama")
        s = OllamaSummarizer(cfg)

    mock_popen.assert_called_once_with(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert s._owned_process is mock_proc


def test_ensure_ollama_running_noop_when_binary_missing():
    """FileNotFoundError (ollama not in PATH) logs an error and leaves _owned_process None."""
    with patch("urllib.request.urlopen", side_effect=OSError("refused")), \
         patch("note_assistant.summarizer.subprocess.Popen", side_effect=FileNotFoundError), \
         patch.object(OllamaSummarizer, "_load"), \
         patch("note_assistant.summarizer._ollama_context_length", return_value=4096):
        cfg = SummarizationConfig(backend="ollama")
        s = OllamaSummarizer(cfg)
    assert s._owned_process is None


def test_ollama_summarizer_shutdown_terminates_owned_process():
    """shutdown() calls terminate() then wait() on the owned process."""
    mock_proc = MagicMock()
    mock_proc.wait.return_value = 0
    with patch.object(OllamaSummarizer, "_load"), \
         patch.object(OllamaSummarizer, "_ensure_ollama_running"), \
         patch("note_assistant.summarizer._ollama_context_length", return_value=4096):
        cfg = SummarizationConfig(backend="ollama")
        s = OllamaSummarizer(cfg)
    s._owned_process = mock_proc
    s.shutdown()
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once_with(timeout=5)
    assert s._owned_process is None


def test_ollama_summarizer_shutdown_kills_when_terminate_times_out():
    """shutdown() escalates to kill() if terminate() times out."""
    mock_proc = MagicMock()
    mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="ollama serve", timeout=5)
    with patch.object(OllamaSummarizer, "_load"), \
         patch.object(OllamaSummarizer, "_ensure_ollama_running"), \
         patch("note_assistant.summarizer._ollama_context_length", return_value=4096):
        cfg = SummarizationConfig(backend="ollama")
        s = OllamaSummarizer(cfg)
    s._owned_process = mock_proc
    s.shutdown()
    mock_proc.kill.assert_called_once()
    assert s._owned_process is None


def test_ollama_summarizer_close_unloads_model_via_keep_alive():
    """close() sends keep_alive=0 to unload the model from the Ollama server."""
    import json
    with patch.object(OllamaSummarizer, "_load"), \
         patch.object(OllamaSummarizer, "_ensure_ollama_running"), \
         patch("note_assistant.summarizer._ollama_context_length", return_value=4096):
        cfg = SummarizationConfig(backend="ollama", ollama_model="qwen3:8b")
        s = OllamaSummarizer(cfg)
    with patch("urllib.request.urlopen") as mock_urlopen:
        s.close()
    mock_urlopen.assert_called_once()
    request_arg = mock_urlopen.call_args[0][0]
    payload = json.loads(request_arg.data)
    assert payload["model"] == "qwen3:8b"
    assert payload["keep_alive"] == 0


def test_ollama_summarizer_shutdown_noop_when_not_owned():
    """shutdown() does nothing when _owned_process is None."""
    with patch.object(OllamaSummarizer, "_load"), \
         patch.object(OllamaSummarizer, "_ensure_ollama_running"), \
         patch("note_assistant.summarizer._ollama_context_length", return_value=4096):
        cfg = SummarizationConfig(backend="ollama")
        s = OllamaSummarizer(cfg)
    s.shutdown()  # must not raise
