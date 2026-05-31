"""CLI entry point — `python -m note_assistant` or `note-assistant` command."""
from __future__ import annotations

import os
import signal
import sys
import threading
from pathlib import Path
from typing import Annotated, Optional

import typer

from .config import AppConfig, detect_best_backends, load_config
from .audio_capture import list_devices

app = typer.Typer(
    name="note-assistant",
    help="Live audio transcription + summarization → Apple Notes",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config_file: Annotated[Optional[Path], typer.Option("--config", "-c",
        help="Path to config.yaml")] = Path("config.yaml"),
    source: Annotated[Optional[str], typer.Option("--source", "-s",
        help="Audio source: mic | system")] = None,
    transcription_backend: Annotated[Optional[str], typer.Option("--transcription",
        help="Transcription backend: apple | faster-whisper | mlx-whisper")] = None,
    whisper_model: Annotated[Optional[str], typer.Option("--whisper-model",
        help="faster-whisper model: tiny|base|small|medium|large-v3")] = None,
    summarization_backend: Annotated[Optional[str], typer.Option("--summarization",
        help="Summarization backend: apple | ollama")] = None,
    ollama_model: Annotated[Optional[str], typer.Option("--ollama-model",
        help="Ollama model name")] = None,
    chunk_seconds: Annotated[Optional[float], typer.Option("--chunk",
        help="Audio chunk size in seconds")] = None,
    no_notes: Annotated[bool, typer.Option("--no-notes",
        help="Disable Apple Notes output")] = False,
    auto: Annotated[bool, typer.Option("--auto",
        help="Auto-detect best backends for this machine")] = False,
    log_level: Annotated[Optional[str], typer.Option("--log-level",
        help="Logging level: DEBUG | INFO | WARNING | ERROR")] = None,
) -> None:
    """Launch the Note Assistant."""
    if ctx.invoked_subcommand is not None:
        return

    cfg = load_config(
        path=config_file,
        source=source,
        transcription_backend=transcription_backend,
        whisper_model=whisper_model,
        summarization_backend=summarization_backend,
        ollama_model=ollama_model,
        chunk_seconds=chunk_seconds,
        log_level=log_level,
    )

    import logging
    effective_level = (log_level or cfg.log_level).upper()
    logging.basicConfig(
        level=effective_level,
        filename="note_assistant.log",
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if no_notes:
        cfg.output.apple_notes = False

    if auto:
        t_backend, s_backend = detect_best_backends()
        cfg.transcription.backend = t_backend  # type: ignore[assignment]
        cfg.summarization.backend = s_backend  # type: ignore[assignment]
        typer.echo(f"✓ Auto-detected: transcription={t_backend}, summarization={s_backend}")

    _launch(cfg)


@app.command("devices")
def list_audio_devices() -> None:
    """List available audio input devices."""
    list_devices()


def _launch(config: AppConfig) -> None:
    """Start UI first, then start pipeline after config confirmed."""
    # Pre-start the multiprocessing resource tracker before Textual opens
    # asyncio/kqueue file descriptors. tqdm (used by huggingface_hub during
    # model download) creates a multiprocessing.RLock, which triggers
    # resource_tracker to spawn a subprocess via spawnv_passfds. If Textual's
    # non-inheritable kqueue fds are already open at that point, the spawn
    # fails with "bad value(s) in fds_to_keep". Initialising here ensures the
    # tracker is already running before those fds exist.
    try:
        import multiprocessing
        _dummy_lock = multiprocessing.RLock()
        del _dummy_lock
    except Exception:
        pass

    from .app import NoteAssistantApp
    from .ui import NoteAssistantUI

    pipeline: NoteAssistantApp | None = None
    pipeline_thread: threading.Thread | None = None

    def start_pipeline(updated_config: AppConfig) -> None:
        nonlocal pipeline, pipeline_thread

        def on_transcript(text: str) -> None:
            ui.call_from_thread(ui.push_transcript, text)

        def on_summary_start() -> None:
            ui.call_from_thread(ui.push_summary_start)

        def on_summary(token: str) -> None:
            ui.call_from_thread(ui.push_summary_token, token)

        def on_chunk() -> None:
            ui.call_from_thread(ui.on_audio_chunk)

        def on_error(source: str, message: str, severity: str) -> None:
            ui.call_from_thread(ui.push_error, source, message, severity)

        def on_progress(current: int, total: int) -> None:
            ui.call_from_thread(ui.push_progress, current, total)

        if updated_config.transcription.backend == "faster-whisper":
            ui.call_from_thread(
                ui.push_error,
                "pipeline",
                f"Loading Whisper '{updated_config.transcription.whisper_model}' model — transcription will start once ready…",
                "warning",
            )
        elif updated_config.transcription.backend == "mlx-whisper":
            ui.call_from_thread(
                ui.push_error,
                "pipeline",
                f"Loading MLX Whisper '{updated_config.transcription.mlx_whisper_model}' model — transcription will start once ready…",
                "warning",
            )
        try:
            pipeline = NoteAssistantApp(
                updated_config,
                on_transcript=on_transcript,
                on_summary=on_summary,
                on_summary_start=on_summary_start,
                on_chunk=on_chunk,
                on_error=on_error,
                on_progress=on_progress,
            )
        except Exception as e:
            ui.call_from_thread(ui.push_error, "pipeline", str(e), "error")
            return
        ui.call_from_thread(ui.set_pipeline, pipeline)
        pipeline_thread = threading.Thread(target=pipeline.run, daemon=False)
        pipeline_thread.start()

    def start_pipeline_async(updated_config: AppConfig) -> None:
        threading.Thread(target=start_pipeline, args=(updated_config,), daemon=True).start()

    ui = NoteAssistantUI(config, on_start_pipeline=start_pipeline_async)

    try:
        ui.run()
    finally:
        if pipeline:
            pipeline.stop()
        if pipeline_thread:
            pipeline_thread.join(timeout=30)

    if getattr(ui, "return_code", None) == 99:
        import subprocess
        # 1. Kill the parent shell with SIGKILL (can't be ignored) so the
        #    launcher's "; exec bash" never runs — otherwise the terminal stays open.
        # 2. Launch osascript detached to close the Terminal.app window after
        #    Python exits. start_new_session puts it in its own process group so
        #    it is unaffected by the parent-shell kill.
        subprocess.Popen(
            ["osascript", "-e", 'tell application "Terminal" to close front window'],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.kill(os.getppid(), signal.SIGKILL)
        sys.exit(0)


if __name__ == "__main__":
    app()
