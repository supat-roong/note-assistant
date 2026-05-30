"""Textual TUI for Note Assistant — Enhanced with File Source and Language Selection."""
from __future__ import annotations

import platform
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal, ScrollableContainer
from textual.widgets import Header, Footer, Label, Button, RichLog, Select, Static, Input

from .config import AppConfig


class StatusBar(Static):
    def update_status(self, source: str, t_backend: str, s_backend: str,
                      chunks: int, is_capturing: bool) -> None:
        heartbeat = "🎤 [green]Capturing...[/]" if is_capturing else "🎤 [yellow]Idle[/]"
        self.update(
            f"Src: [cyan]{source}[/] | "
            f"T: [cyan]{t_backend}[/] | S: [cyan]{s_backend}[/] | "
            f"Chunks: [bold]{chunks}[/] | {heartbeat}"
        )


class NoteAssistantUI(App):
    """Main Textual application with multi-language and file source support."""

    BINDINGS = [
        Binding("ctrl+p", "toggle_pause", "Pause/Resume", priority=True),
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
    #file-path {
        margin-top: 1;
        display: none;
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
                yield Input(
                    placeholder="Enter absolute path to audio file...",
                    id="file-path",
                    value=str(self._config.audio.file_path or "")
                )
            
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
                with Horizontal():
                    yield Select([("Apple Speech", "apple"), ("Whisper", "faster-whisper")], value=self._config.transcription.backend, id="t-backend")
                    yield Select([("Apple Intelligence", "apple"), ("MLX (on-device)", "mlx"), ("Ollama", "ollama")], value=self._config.summarization.backend, id="s-backend")
                yield Label("", id="summarization-status", classes="status-label")
            
            yield Button("🚀 Start Processing", variant="success", id="start-btn")

        # Recording View
        with Horizontal(id="recording-view"):
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
        # Initial visibility
        self._update_file_input_visibility(self._config.audio.source)

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

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "audio-source":
            self._update_file_input_visibility(event.value)

    def _update_file_input_visibility(self, source: str) -> None:
        field = self.query_one("#file-path", Input)
        field.display = (source == "file")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-btn":
            # Update config
            self._config.audio.source = self.query_one("#audio-source", Select).value
            if self._config.audio.source == "file":
                path_str = self.query_one("#file-path", Input).value
                if not path_str:
                    self.notify("Error: Please provide a file path", severity="error")
                    return
                self._config.audio.file_path = Path(path_str)
            
            self._config.language_input = self.query_one("#lang-input", Select).value
            self._config.language_output = self.query_one("#lang-output", Select).value
            self._config.transcription.backend = self.query_one("#t-backend", Select).value
            self._config.summarization.backend = self.query_one("#s-backend", Select).value
            
            self.query_one("#settings-view").display = False
            self.query_one("#recording-view").display = True
            self._start_pipeline(self._config)

    def on_audio_chunk(self) -> None:
        self._last_chunk_at = datetime.now()
        self._update_status_bar()

    def push_transcript(self, text: str) -> None:
        if self._paused: return
        self._chunk_count += 1
        self.query_one("#transcript-log", RichLog).write(text + " ")
        self._update_status_bar()

    def push_summary_token(self, token: str) -> None:
        self.query_one("#summary-log", RichLog).write(token, scroll_end=True)

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
            self._config.summarization.backend, self._chunk_count, is_capturing
        )
