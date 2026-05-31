import subprocess
import pytest
from unittest.mock import MagicMock, PropertyMock, patch
from textual.widgets import Button, Input, Label, ProgressBar
from note_assistant.ui import NoteAssistantUI, StatusBar, _run_file_picker
from note_assistant.config import AppConfig, AudioConfig, TranscriptionConfig, SummarizationConfig, OutputConfig


def test_run_file_picker_returns_path_on_success():
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "/Users/bob/audio.wav\n"
    with patch("subprocess.run", return_value=mock_result):
        assert _run_file_picker() == "/Users/bob/audio.wav"


def test_run_file_picker_returns_none_on_cancel():
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "User canceled."
    with patch("subprocess.run", return_value=mock_result):
        assert _run_file_picker() is None


def test_run_file_picker_raises_on_osascript_missing():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match="osascript"):
            _run_file_picker()


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
        row = pilot.app.query_one("#file-path-row")
        assert not row.display


async def test_file_path_input_visible_for_file_source(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        pilot.app._update_file_input_visibility("file")
        await pilot.pause()
        assert pilot.app.query_one("#file-path-row").display


async def test_stop_button_present_in_recording_view(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        btn = pilot.app.query_one("#start-btn", Button)
        pilot.app.post_message(Button.Pressed(btn))
        await pilot.pause()
        assert pilot.app.query_one("#stop-btn") is not None


async def test_elapsed_label_updates_when_pipeline_set(ui_config):
    pipeline_mock = MagicMock()
    type(pipeline_mock).elapsed = PropertyMock(return_value="01:23")
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        btn = pilot.app.query_one("#start-btn", Button)
        pilot.app.post_message(Button.Pressed(btn))
        await pilot.pause()
        pilot.app.set_pipeline(pipeline_mock)
        pilot.app._tick_elapsed()
        await pilot.pause()
        label = pilot.app.query_one("#elapsed-label", Label)
        assert str(label._Static__content) == "01:23"


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


async def test_settings_all_sections_visible(ui_config):
    """Regression: all setting groups must have non-zero height (CSS layout fix)."""
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 90)) as pilot:
        for widget in pilot.app.query(".setting-group"):
            assert widget.region.height > 4, (
                f"setting-group too small: h={widget.region.height} — CSS height:auto regression"
            )
        start = pilot.app.query_one("#start-btn")
        assert start.region.y < 90, f"start-btn off-screen at y={start.region.y}"


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


async def test_browse_button_populates_file_path_on_success(ui_config, tmp_path):
    fake_path = str(tmp_path / "audio.wav")
    (tmp_path / "audio.wav").touch()
    with patch("note_assistant.ui._run_file_picker", return_value=fake_path):
        async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
            pilot.app._update_file_input_visibility("file")
            await pilot.pause()
            btn = pilot.app.query_one("#browse-btn", Button)
            pilot.app.post_message(Button.Pressed(btn))
            await pilot.pause(delay=0.5)
            assert pilot.app.query_one("#file-path", Input).value == fake_path


async def test_browse_button_cancel_leaves_path_unchanged(ui_config):
    with patch("note_assistant.ui._run_file_picker", return_value=None):
        async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
            pilot.app._update_file_input_visibility("file")
            await pilot.pause()
            pilot.app.query_one("#file-path", Input).value = "/existing/path.wav"
            btn = pilot.app.query_one("#browse-btn", Button)
            pilot.app.post_message(Button.Pressed(btn))
            await pilot.pause(delay=0.5)
            assert pilot.app.query_one("#file-path", Input).value == "/existing/path.wav"


async def test_browse_button_notifies_on_osascript_error(ui_config):
    from textual.worker import WorkerState
    notified = []
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        pilot.app.notify = lambda msg, **kw: notified.append(msg)
        mock_worker = MagicMock()
        mock_worker.name = "browse-file"
        mock_event = MagicMock()
        mock_event.worker = mock_worker
        mock_event.state = WorkerState.ERROR
        pilot.app.on_worker_state_changed(mock_event)
        await pilot.pause()
        assert any("picker" in n.lower() or "unavailable" in n.lower() for n in notified)


async def test_file_progress_bar_hidden_by_default(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        assert not pilot.app.query_one("#file-progress").display


async def test_file_progress_bar_shown_for_file_source(ui_config):
    ui_config.audio.source = "file"
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        pilot.app.query_one("#file-path", Input).value = "/dev/null"
        btn = pilot.app.query_one("#start-btn", Button)
        pilot.app.post_message(Button.Pressed(btn))
        await pilot.pause()
        assert pilot.app.query_one("#file-progress").display


async def test_push_progress_updates_bar(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        pilot.app.push_progress(3, 10)
        await pilot.pause()
        bar = pilot.app.query_one("#file-progress", ProgressBar)
        assert bar.progress == 3


async def test_push_progress_exits_at_100(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        pilot.app.push_progress(10, 10)
        await pilot.pause(delay=1.5)
        assert pilot.app.query_one("#done-view").display


async def test_auto_title_switch_renders_with_default_on(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 80)) as pilot:
        from textual.widgets import Switch
        switch = pilot.app.query_one("#auto-title", Switch)
        assert switch.value is True


async def test_start_button_reads_auto_title_switch_off(ui_config):
    from textual.widgets import Switch
    received = []
    async with NoteAssistantUI(ui_config, on_start_pipeline=received.append).run_test(size=(120, 80)) as pilot:
        pilot.app.query_one("#auto-title", Switch).value = False
        await pilot.pause()
        btn = pilot.app.query_one("#start-btn", Button)
        pilot.app.post_message(Button.Pressed(btn))
        await pilot.pause()
        assert len(received) == 1
        assert received[0].output.auto_title is False


async def test_start_button_reads_auto_title_switch_on(ui_config):
    from textual.widgets import Switch
    received = []
    async with NoteAssistantUI(ui_config, on_start_pipeline=received.append).run_test(size=(120, 80)) as pilot:
        pilot.app.query_one("#auto-title", Switch).value = True
        await pilot.pause()
        btn = pilot.app.query_one("#start-btn", Button)
        pilot.app.post_message(Button.Pressed(btn))
        await pilot.pause()
        assert received[0].output.auto_title is True


async def test_save_recording_switch_renders_with_default_off(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 80)) as pilot:
        from textual.widgets import Switch
        switch = pilot.app.query_one("#save-recording", Switch)
        assert switch.value is False


async def test_start_button_reads_save_recording_switch_on(ui_config):
    from textual.widgets import Switch
    received = []
    async with NoteAssistantUI(ui_config, on_start_pipeline=received.append).run_test(size=(120, 80)) as pilot:
        pilot.app.query_one("#save-recording", Switch).value = True
        await pilot.pause()
        btn = pilot.app.query_one("#start-btn", Button)
        pilot.app.post_message(Button.Pressed(btn))
        await pilot.pause()
        assert len(received) == 1
        assert received[0].output.save_recording is True


async def test_start_button_reads_save_recording_switch_off(ui_config):
    from textual.widgets import Switch
    received = []
    async with NoteAssistantUI(ui_config, on_start_pipeline=received.append).run_test(size=(120, 80)) as pilot:
        pilot.app.query_one("#save-recording", Switch).value = False
        await pilot.pause()
        btn = pilot.app.query_one("#start-btn", Button)
        pilot.app.post_message(Button.Pressed(btn))
        await pilot.pause()
        assert received[0].output.save_recording is False


async def test_done_view_hidden_by_default(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        assert not pilot.app.query_one("#done-view").display


async def test_done_view_has_restart_and_quit_buttons(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        assert pilot.app.query_one("#restart-btn") is not None
        assert pilot.app.query_one("#quit-btn") is not None


async def test_show_done_view_hides_recording_shows_done(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        pilot.app.query_one("#settings-view").display = False
        pilot.app.query_one("#recording-view").display = True
        pilot.app._show_done_view()
        await pilot.pause()
        assert not pilot.app.query_one("#recording-view").display
        assert pilot.app.query_one("#done-view").display


async def test_stop_button_shows_done_view_not_exit(ui_config):
    exited = []
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        pilot.app.exit = lambda *a, **kw: exited.append(True)
        btn = pilot.app.query_one("#start-btn", Button)
        pilot.app.post_message(Button.Pressed(btn))
        await pilot.pause()
        stop_btn = pilot.app.query_one("#stop-btn", Button)
        pilot.app.post_message(Button.Pressed(stop_btn))
        await pilot.pause()
        assert not exited, "exit() must NOT be called — done-view should show instead"
        assert pilot.app.query_one("#done-view").display


async def test_restart_button_shows_settings_view(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        pilot.app.query_one("#recording-view").display = True
        pilot.app.query_one("#settings-view").display = False
        pilot.app._show_done_view()
        await pilot.pause()
        btn = pilot.app.query_one("#restart-btn", Button)
        pilot.app.post_message(Button.Pressed(btn))
        await pilot.pause()
        assert pilot.app.query_one("#settings-view").display
        assert not pilot.app.query_one("#done-view").display


async def test_restart_resets_internal_state(ui_config):
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        pilot.app._chunk_count = 5
        pilot.app._paused = True
        pilot.app._pipeline = MagicMock()
        pilot.app.query_one("#recording-view").display = True
        pilot.app.query_one("#settings-view").display = False
        pilot.app._show_done_view()
        await pilot.pause()
        btn = pilot.app.query_one("#restart-btn", Button)
        pilot.app.post_message(Button.Pressed(btn))
        await pilot.pause()
        assert pilot.app._chunk_count == 0
        assert pilot.app._paused is False
        assert pilot.app._pipeline is None
        assert pilot.app._last_chunk_at is None


async def test_push_progress_at_100_shows_done_view(ui_config):
    exited = []
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        pilot.app.exit = lambda *a, **kw: exited.append(True)
        pilot.app.push_progress(10, 10)
        await pilot.pause(delay=1.5)
        assert not exited, "exit() must NOT be called — done-view should show instead"
        assert pilot.app.query_one("#done-view").display


async def test_quit_button_calls_exit_with_code_99(ui_config):
    exit_calls = []
    async with NoteAssistantUI(ui_config, on_start_pipeline=lambda c: None).run_test(size=(120, 70)) as pilot:
        pilot.app.exit = lambda *a, **kw: exit_calls.append((a, kw))
        btn = pilot.app.query_one("#quit-btn", Button)
        pilot.app.post_message(Button.Pressed(btn))
        await pilot.pause()
        assert exit_calls, "exit() was not called"
        args, kwargs = exit_calls[0]
        assert kwargs.get("return_code") == 99 or (args and args[0] == 99)
