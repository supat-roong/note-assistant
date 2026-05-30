"""Textual TUI for Note Assistant — Enhanced with File Source and Language Selection."""
from __future__ import annotations

import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal, ScrollableContainer
from textual.widgets import Header, Footer, Label, Button, ProgressBar, RichLog, Select, Static, Input, Switch
from textual.worker import WorkerState

from .config import AppConfig


def _run_file_picker() -> str | None:
    script = (
        'set f to choose file with prompt "Select audio or video file:" '
        'of type {"wav", "mp3", "m4a", "aiff", "ogg", "flac",'
        ' "mp4", "mov", "avi", "mkv", "m4v", "webm", "wmv", "3gp", "ts", "flv"}\n'
        'POSIX path of f'
    )
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("osascript not available on this system")
    if result.returncode == 0:
        return result.stdout.strip()
    if result.returncode == 1:
        return None  # user pressed Cancel in the dialog
    raise RuntimeError(f"osascript error: {result.stderr.strip()}")


class StatusBar(Static):
    def update_status(self, source: str, t_backend: str, s_backend: str,
                      chunks: int, is_capturing: bool, is_paused: bool = False) -> None:
        if is_paused:
            heartbeat = "⏸ [yellow]Paused[/]"
        elif is_capturing:
            heartbeat = "🎤 [green]Capturing...[/]"
        else:
            heartbeat = "🎤 [dim]Idle[/]"
        self.update(
            f"Src: [cyan]{source}[/] | "
            f"T: [cyan]{t_backend}[/] | S: [cyan]{s_backend}[/] | "
            f"Chunks: [bold]{chunks}[/] | {heartbeat}"
        )


class NoteAssistantUI(App):
    """Main Textual application with multi-language and file source support."""

    BINDINGS = [
        Binding("ctrl+p", "toggle_pause", "Pause/Resume", priority=True),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    CSS = """
    #settings-view {
        padding: 1 2;
        background: $surface;
        height: auto;
        overflow-y: auto;
    }
    #recording-view {
        display: none;
        height: 1fr;
    }
    .setting-group {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        border: solid $accent;
    }
    .status-label {
        color: $text-muted;
        margin-bottom: 1;
        text-style: italic;
    }
    #file-path-row {
        margin-top: 1;
        display: none;
        height: auto;
    }
    #file-path-row Input {
        width: 1fr;
    }
    #lang-row {
        height: auto;
        margin-bottom: 1;
    }
    #lang-row > Vertical {
        height: auto;
        width: 1fr;
        padding: 0 1;
    }
    .setting-group > Horizontal {
        height: auto;
    }
    #backends-row > Vertical {
        height: auto;
        width: 1fr;
        padding: 0 1;
    }
    #auto-title-row {
        height: auto;
    }
    #auto-title-row Label {
        width: 1fr;
        content-align: left middle;
    }
    #stop-row {
        height: auto;
        padding: 0 1;
    }
    #file-progress {
        display: none;
        margin: 0 1 1 1;
        height: auto;
    }
    #elapsed-label {
        margin-left: 2;
        height: 1;
        content-align: left middle;
    }
    #panels-row {
        height: 1fr;
    }
    #transcript-panel {
        width: 1fr;
        border: solid $accent;
    }
    #summary-panel {
        width: 1fr;
        border: solid $success;
    }
    StatusBar {
        dock: bottom;
        height: 1;
        background: $boost;
    }
    """

    def __init__(self, config: AppConfig, on_start_pipeline: Callable[[AppConfig], None]):
        super().__init__()
        self._config = config
        self._start_pipeline = on_start_pipeline
        self._paused = False
        self._chunk_count = 0
        self._last_chunk_at: Optional[datetime] = None
        self._pipeline = None

    def set_pipeline(self, app) -> None:
        self._pipeline = app

    def compose(self) -> ComposeResult:
        yield Header()
        
        # Settings View
        with Vertical(id="settings-view"):
            yield Label("⚙️  Note Assistant Settings", id="settings-title")
            
            with Vertical(classes="setting-group"):
                yield Label("Audio Source")
                yield Select(
                    [("Microphone", "mic"), ("System Audio", "system"), ("Audio File", "file")],
                    value=self._config.audio.source,
                    id="audio-source"
                )
                with Horizontal(id="file-path-row"):
                    yield Input(
                        placeholder="Enter absolute path to audio file...",
                        id="file-path",
                        value=str(self._config.audio.file_path or "")
                    )
                    yield Button("Browse...", id="browse-btn")
            
            with Horizontal(id="lang-row"):
                with Vertical():
                    yield Label("Input Language")
                    yield Select(
                        [("English", "English"), ("Thai", "Thai"), ("Japanese", "Japanese"), ("Chinese", "Chinese"), ("Auto-detect", "Auto")],
                        value=self._config.language_input,
                        id="lang-input"
                    )
                with Vertical():
                    yield Label("Output Language (Summary)")
                    yield Select(
                        [("English", "English"), ("Thai", "Thai"), ("Japanese", "Japanese"), ("Chinese", "Chinese")],
                        value=self._config.language_output,
                        id="lang-output"
                    )

            with Vertical(classes="setting-group"):
                yield Label("Backends")
                with Horizontal(id="backends-row"):
                    with Vertical():
                        yield Label("Transcription")
                        yield Select(
                            [("Apple Speech", "apple"), ("Whisper", "faster-whisper")],
                            value=self._config.transcription.backend,
                            id="t-backend",
                        )
                    with Vertical():
                        yield Label("Summarization")
                        yield Select(
                            [("Apple Intelligence", "apple"), ("MLX (on-device)", "mlx"), ("Ollama", "ollama")],
                            value=self._config.summarization.backend,
                            id="s-backend",
                        )
                yield Label("", id="summarization-status", classes="status-label")

            with Vertical(classes="setting-group"):
                yield Label("Output")
                with Horizontal(id="auto-title-row"):
                    yield Label("Auto-generate note title")
                    yield Switch(value=self._config.output.auto_title, id="auto-title")

            yield Button("🚀 Start Processing", variant="success", id="start-btn")

        # Recording View
        with Vertical(id="recording-view"):
            with Horizontal(id="stop-row"):
                yield Button("⏹ Stop", variant="error", id="stop-btn")
                yield Label("00:00", id="elapsed-label")
            yield ProgressBar(id="file-progress", total=100, show_eta=False)
            with Horizontal(id="panels-row"):
                with ScrollableContainer(id="transcript-panel"):
                    yield Label("📝 Transcript")
                    yield RichLog(id="transcript-log", wrap=True)
                with ScrollableContainer(id="summary-panel"):
                    yield Label("✨ Summary")
                    yield RichLog(id="summary-log", wrap=True)

        yield StatusBar(id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self._check_apple_intelligence()
        self._update_file_input_visibility(self._config.audio.source)
        self._update_transcription_for_lang(self._config.language_input)
        self.set_interval(1.0, self._tick_elapsed)

    def _tick_elapsed(self) -> None:
        if self._pipeline is None:
            return
        try:
            self.query_one("#elapsed-label", Label).update(self._pipeline.elapsed)
        except Exception:
            pass

    def _check_apple_intelligence(self) -> None:
        try:
            label = self.query_one("#summarization-status", Label)
            ver_str = platform.mac_ver()[0]
            major = int(ver_str.split(".")[0]) if ver_str else 0
            bridge = Path.home() / ".note-assistant" / "foundation_model_bridge"
            if major >= 26 and bridge.exists():
                label.update("✅ Apple Foundation Models ready!")
            elif major >= 26:
                label.update("⚠️  Run setup_mac.sh to build the Foundation Models bridge")
            else:
                label.update("⚠️  Apple Foundation Models requires macOS 26+ (Xcode 26 SDK)")
        except Exception:
            pass

    _LANG_TO_WHISPER: dict[str, str | None] = {
        "English": "en",
        "Thai": "th",
        "Japanese": "ja",
        "Chinese": "zh",
        "Auto": None,
    }

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "audio-source":
            self._update_file_input_visibility(event.value)
        elif event.select.id == "lang-input":
            self._update_transcription_for_lang(event.value)

    def _update_transcription_for_lang(self, lang: str) -> None:
        t_backend = self.query_one("#t-backend", Select)
        if lang == "Auto":
            t_backend.value = "faster-whisper"
            t_backend.disabled = True
        else:
            t_backend.disabled = False

    def _update_file_input_visibility(self, source: str) -> None:
        row = self.query_one("#file-path-row")
        row.display = (source == "file")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "stop-btn":
            self.exit()
        elif event.button.id == "browse-btn":
            self.run_worker(_run_file_picker, thread=True, name="browse-file")
        elif event.button.id == "start-btn":
            # Update config
            self._config.audio.source = self.query_one("#audio-source", Select).value
            if self._config.audio.source == "file":
                path_str = self.query_one("#file-path", Input).value
                if not path_str:
                    self.notify("Error: Please provide a file path", severity="error")
                    return
                self._config.audio.file_path = Path(path_str)
                if not self._config.audio.file_path.exists():
                    self.notify("Error: File not found", severity="error")
                    return

            self._config.language_input = self.query_one("#lang-input", Select).value
            self._config.language_output = self.query_one("#lang-output", Select).value
            self._config.transcription.language = self._LANG_TO_WHISPER.get(self._config.language_input)
            if self._config.language_input == "Auto":
                self._config.transcription.backend = "faster-whisper"
            else:
                self._config.transcription.backend = self.query_one("#t-backend", Select).value
            self._config.summarization.backend = self.query_one("#s-backend", Select).value
            self._config.output.auto_title = self.query_one("#auto-title", Switch).value

            self.query_one("#settings-view").display = False
            self.query_one("#recording-view").display = True
            self.query_one("#file-progress").display = (self._config.audio.source == "file")
            self._update_status_bar()
            self._start_pipeline(self._config)

    def on_worker_state_changed(self, event) -> None:
        if event.worker.name != "browse-file":
            return
        if event.state == WorkerState.SUCCESS:
            path = event.worker.result
            if path:
                self.query_one("#file-path", Input).value = path
        elif event.state == WorkerState.ERROR:
            self.notify("File picker unavailable — enter path manually", severity="warning")

    def on_audio_chunk(self) -> None:
        self._last_chunk_at = datetime.now()
        self._update_status_bar()

    def push_progress(self, current: int, total: int) -> None:
        bar = self.query_one("#file-progress", ProgressBar)
        bar.update(total=total, progress=current)
        if total > 0 and current >= total:
            self.set_timer(1.0, self.exit)

    def push_transcript(self, text: str) -> None:
        if self._paused: return
        self._chunk_count += 1
        self.query_one("#transcript-log", RichLog).write(text + " ")
        self._update_status_bar()

    def push_summary_token(self, text: str) -> None:
        log = self.query_one("#summary-log", RichLog)
        log.clear()
        log.write(text, scroll_end=True)

    def push_error(self, source: str, message: str, severity: str = "error") -> None:
        self.notify(f"[{source}] {message}", severity=severity)

    def action_toggle_pause(self) -> None:
        if self._pipeline is None:
            return
        if self._paused:
            self._pipeline.resume()
        else:
            self._pipeline.pause()
        self._paused = not self._paused
        self._update_status_bar()

    def _update_status_bar(self) -> None:
        is_capturing = False
        if self._last_chunk_at:
            is_capturing = (datetime.now() - self._last_chunk_at).total_seconds() < 4.0

        self.query_one(StatusBar).update_status(
            self._config.audio.source, self._config.transcription.backend,
            self._config.summarization.backend, self._chunk_count,
            is_capturing=is_capturing, is_paused=self._paused,
        )
