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

    ENABLE_COMMAND_PALETTE = False

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
    #done-view {
        display: none;
        align: center middle;
        height: 1fr;
    }
    #done-label {
        text-align: center;
        margin-bottom: 2;
    }
    #done-buttons {
        height: auto;
    }
    #done-buttons Button {
        margin: 0 1;
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
        self._summary_buf = ""
        self._t_loaded = True
        self._s_loaded = True
        self._t_model_label = ""
        self._s_model_label = ""

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
                        t_options = [("Apple Speech", "apple")]
                        if platform.machine() == "arm64":
                            t_options.append(("Whisper (MLX)", "mlx-whisper"))
                        t_options.append(("Whisper (CPU)", "faster-whisper"))
                        yield Select(
                            t_options,
                            value=self._config.transcription.backend,
                            id="t-backend",
                        )
                    with Vertical():
                        yield Label("Summarization")
                        mlx_short = self._config.summarization.mlx_model.split("/")[-1]
                        ollama_model = self._config.summarization.ollama_model
                        yield Select(
                            [
                                ("Apple Intelligence", "apple"),
                                (f"{mlx_short} (mlx)", "mlx"),
                                (f"{ollama_model} (ollama)", "ollama"),
                            ],
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
                    yield Label("📝 Transcript", id="transcript-panel-title")
                    yield RichLog(id="transcript-log", wrap=True)
                with ScrollableContainer(id="summary-panel"):
                    yield Label("✨ Summary", id="summary-panel-title")
                    yield RichLog(id="summary-log", wrap=True)

        # Done View
        with Vertical(id="done-view"):
            yield Label("✅ Recording stopped.", id="done-label")
            with Horizontal(id="done-buttons"):
                yield Button("🔄 Restart", variant="primary", id="restart-btn")
                yield Button("✖ Quit", variant="error", id="quit-btn")

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
        whisper_options = []
        if platform.machine() == "arm64":
            whisper_options.append(("Whisper (MLX)", "mlx-whisper"))
        whisper_options.append(("Whisper (CPU)", "faster-whisper"))
        if lang == "Auto":
            # Apple Speech requires a fixed locale — remove it from the dropdown.
            t_backend.set_options(whisper_options)
            if t_backend.value == "apple":
                t_backend.value = "mlx-whisper" if platform.machine() == "arm64" else "faster-whisper"
        else:
            t_backend.set_options([("Apple Speech", "apple")] + whisper_options)

    def _update_file_input_visibility(self, source: str) -> None:
        row = self.query_one("#file-path-row")
        row.display = (source == "file")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "stop-btn":
            self._show_done_view()
        elif event.button.id == "restart-btn":
            self.query_one("#done-view").display = False
            self.query_one("#transcript-log", RichLog).clear()
            self.query_one("#summary-log", RichLog).clear()
            self.query_one("#elapsed-label", Label).update("00:00")
            self.query_one("#file-progress").display = False
            self.query_one("#transcript-panel-title", Label).update("📝 Transcript")
            self.query_one("#summary-panel-title", Label).update("✨ Summary")
            self._chunk_count = 0
            self._last_chunk_at = None
            self._paused = False
            self._pipeline = None
            self._summary_buf = ""
            self.query_one("#settings-view").display = True
        elif event.button.id == "quit-btn":
            self.exit(return_code=99)
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
            chosen_backend = self.query_one("#t-backend", Select).value
            if self._config.language_input == "Auto" and chosen_backend == "apple":
                chosen_backend = "mlx-whisper" if platform.machine() == "arm64" else "faster-whisper"
            self._config.transcription.backend = chosen_backend
            self._config.summarization.backend = self.query_one("#s-backend", Select).value
            self._config.output.auto_title = self.query_one("#auto-title", Switch).value

            self.query_one("#settings-view").display = False
            self.query_one("#recording-view").display = True
            self.query_one("#file-progress").display = (self._config.audio.source == "file")
            self._set_panel_titles()
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
            self.set_timer(1.0, self._show_done_view)

    def _t_model_info(self) -> tuple[str, bool]:
        backend = self._config.transcription.backend
        if backend == "apple":
            return "Apple Speech", False
        elif backend == "mlx-whisper":
            short = self._config.transcription.mlx_whisper_model.split("/")[-1]
            return f"{short} (mlx-whisper)", True
        else:
            return f"{self._config.transcription.whisper_model} (faster-whisper)", True

    def _s_model_info(self) -> tuple[str, bool]:
        backend = self._config.summarization.backend
        if backend == "apple":
            return "Apple Intelligence", False
        elif backend == "mlx":
            short = self._config.summarization.mlx_model.split("/")[-1]
            return f"{short} (mlx)", True
        else:
            return f"{self._config.summarization.ollama_model} (ollama)", True

    def _set_panel_titles(self) -> None:
        t_label, t_loading = self._t_model_info()
        s_label, s_loading = self._s_model_info()
        self._t_model_label = t_label
        self._s_model_label = s_label
        self._t_loaded = not t_loading
        self._s_loaded = not s_loading
        t_text = f"📝 Transcription — {t_label}  ⏳" if t_loading else f"📝 Transcription — {t_label}"
        s_text = f"✨ Summary — {s_label}  ⏳" if s_loading else f"✨ Summary — {s_label}"
        self.query_one("#transcript-panel-title", Label).update(t_text)
        self.query_one("#summary-panel-title", Label).update(s_text)

    def push_transcript(self, text: str) -> None:
        if self._paused: return
        self._chunk_count += 1
        if not self._t_loaded:
            self._t_loaded = True
            self.query_one("#transcript-panel-title", Label).update(f"📝 Transcription — {self._t_model_label}")
        self.query_one("#transcript-log", RichLog).write(text + " ")
        self._update_status_bar()

    def push_summary_start(self) -> None:
        if not self._s_loaded:
            self._s_loaded = True
            self.query_one("#summary-panel-title", Label).update(f"✨ Summary — {self._s_model_label}")
        self._summary_buf = ""
        self.query_one("#summary-log", RichLog).clear()

    def push_summary_token(self, text: str) -> None:
        self._summary_buf += text
        log = self.query_one("#summary-log", RichLog)
        log.clear()
        log.write(self._summary_buf, scroll_end=True)

    def push_error(self, source: str, message: str, severity: str = "error") -> None:
        self.notify(f"[{source}] {message}", severity=severity)

    def push_backend_switch(self, model_label: str) -> None:
        self._s_model_label = model_label
        self._s_loaded = True
        self.query_one("#summary-panel-title", Label).update(f"✨ Summary — {model_label}")
        self.notify(f"Switched to fallback: {model_label}", severity="warning")

    def action_toggle_pause(self) -> None:
        if self._pipeline is None:
            return
        if self._paused:
            self._pipeline.resume()
        else:
            self._pipeline.pause()
        self._paused = not self._paused
        self._update_status_bar()

    def _show_done_view(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self.query_one("#recording-view").display = False
        self.query_one("#done-view").display = True

    def _update_status_bar(self) -> None:
        is_capturing = False
        if self._last_chunk_at:
            is_capturing = (datetime.now() - self._last_chunk_at).total_seconds() < 4.0

        self.query_one(StatusBar).update_status(
            self._config.audio.source, self._config.transcription.backend,
            self._config.summarization.backend, self._chunk_count,
            is_capturing=is_capturing, is_paused=self._paused,
        )
