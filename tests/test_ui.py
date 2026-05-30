import pytest
from unittest.mock import MagicMock
from textual.widgets import Button
from note_assistant.ui import NoteAssistantUI, StatusBar
from note_assistant.config import AppConfig, AudioConfig, TranscriptionConfig, SummarizationConfig, OutputConfig


@pytest.fixture
def ui_config(tmp_path):
    return AppConfig(
        audio=AudioConfig(source="mic"),
        transcription=TranscriptionConfig(backend="faster-whisper"),
        summarization=SummarizationConfig(backend="ollama"),
        output=OutputConfig(
            apple_notes=False,
            save_transcript=False,
            save_summary=False,
            output_dir=tmp_path,
        ),
    )


async def test_settings_screen_renders(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        assert pilot.app.query_one("#settings-view").display
        assert not pilot.app.query_one("#recording-view").display


async def test_start_button_switches_to_recording_view(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        btn = pilot.app.query_one("#start-btn", Button)
        pilot.app.post_message(Button.Pressed(btn))
        await pilot.pause()
        assert not pilot.app.query_one("#settings-view").display
        assert pilot.app.query_one("#recording-view").display


async def test_start_button_passes_config_to_callback(ui_config):
    received = []
    async with NoteAssistantUI(ui_config, on_start_pipeline=received.append).run_test(size=(120, 70)) as pilot:
        btn = pilot.app.query_one("#start-btn", Button)
        pilot.app.post_message(Button.Pressed(btn))
        await pilot.pause()
        assert len(received) == 1
        assert isinstance(received[0], AppConfig)


async def test_push_transcript_writes_to_log(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        btn = pilot.app.query_one("#start-btn", Button)
        pilot.app.post_message(Button.Pressed(btn))
        await pilot.pause()
        pilot.app.push_transcript("hello world")
        await pilot.pause()
        log = pilot.app.query_one("#transcript-log")
        assert log is not None


async def test_push_summary_token_writes_to_log(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        btn = pilot.app.query_one("#start-btn", Button)
        pilot.app.post_message(Button.Pressed(btn))
        await pilot.pause()
        pilot.app.push_summary_token("summary text")
        await pilot.pause()
        log = pilot.app.query_one("#summary-log")
        assert log is not None


async def test_push_error_calls_notify(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        notified = []
        pilot.app.notify = lambda msg, **kw: notified.append(msg)
        pilot.app.push_error("transcriber", "device lost", "error")
        assert any("device lost" in n for n in notified)


async def test_file_path_input_hidden_by_default(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        file_input = pilot.app.query_one("#file-path")
        assert not file_input.display


async def test_file_path_input_visible_for_file_source(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        pilot.app._update_file_input_visibility("file")
        await pilot.pause()
        assert pilot.app.query_one("#file-path").display


async def test_status_bar_shows_paused_when_ctrl_p(ui_config):
    pipeline_mock = MagicMock()
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        btn = pilot.app.query_one("#start-btn", Button)
        pilot.app.post_message(Button.Pressed(btn))
        await pilot.pause()
        pilot.app.set_pipeline(pipeline_mock)
        await pilot.press("ctrl+p")
        await pilot.pause()
        status = pilot.app.query_one(StatusBar)
        assert "Paused" in str(status._Static__content)


async def test_status_bar_shows_idle_when_recording_starts(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        btn = pilot.app.query_one("#start-btn", Button)
        pilot.app.post_message(Button.Pressed(btn))
        await pilot.pause()
        status = pilot.app.query_one(StatusBar)
        assert "Idle" in str(status._Static__content)


async def test_start_with_nonexistent_file_shows_error(ui_config, tmp_path):
    ui_config.audio.source = "file"
    notified = []
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        pilot.app.notify = lambda msg, **kw: notified.append(msg)
        pilot.app._update_file_input_visibility("file")
        await pilot.pause()
        pilot.app.query_one("#file-path").value = str(tmp_path / "missing.wav")
        btn = pilot.app.query_one("#start-btn", Button)
        pilot.app.post_message(Button.Pressed(btn))
        await pilot.pause()
        assert any("not found" in n.lower() for n in notified)
        assert pilot.app.query_one("#settings-view").display


async def test_backend_labels_present(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        label_texts = [str(w._Static__content) for w in pilot.app.query("Label")]
        assert any("Transcription" in t for t in label_texts)
        assert any("Summarization" in t for t in label_texts)


async def test_ctrl_q_binding_exists(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        bindings = {b.key for b in pilot.app.BINDINGS}
        assert "ctrl+q" in bindings


async def test_pause_resume_via_ctrl_p(ui_config):
    pipeline_mock = MagicMock()
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        btn = pilot.app.query_one("#start-btn", Button)
        pilot.app.post_message(Button.Pressed(btn))
        await pilot.pause()
        pilot.app.set_pipeline(pipeline_mock)
        await pilot.press("ctrl+p")
        pipeline_mock.pause.assert_called_once()
        await pilot.press("ctrl+p")
        pipeline_mock.resume.assert_called_once()
