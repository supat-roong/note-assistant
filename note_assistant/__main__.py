"""CLI entry point — `python -m note_assistant` or `note-assistant` command."""
from __future__ import annotations

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
        help="Transcription backend: apple | faster-whisper")] = None,
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
    from .app import NoteAssistantApp
    from .ui import NoteAssistantUI

    pipeline: NoteAssistantApp | None = None
    pipeline_thread: threading.Thread | None = None

    def start_pipeline(updated_config: AppConfig) -> None:
        nonlocal pipeline, pipeline_thread

        def on_transcript(text: str) -> None:
            ui.call_from_thread(ui.push_transcript, text)

        def on_summary(token: str) -> None:
            ui.call_from_thread(ui.push_summary_token, token)

        def on_chunk() -> None:
            ui.call_from_thread(ui.on_audio_chunk)

        def on_error(source: str, message: str, severity: str) -> None:
            ui.call_from_thread(ui.push_error, source, message, severity)

        pipeline = NoteAssistantApp(
            updated_config,
            on_transcript=on_transcript,
            on_summary=on_summary,
            on_chunk=on_chunk,
            on_error=on_error,
        )
        ui.call_from_thread(ui.set_pipeline, pipeline)
        pipeline_thread = threading.Thread(target=pipeline.run, daemon=True)
        pipeline_thread.start()

    ui = NoteAssistantUI(config, on_start_pipeline=start_pipeline)

    try:
        ui.run()
    finally:
        if pipeline:
            pipeline.stop()
        if pipeline_thread:
            pipeline_thread.join(timeout=2)


if __name__ == "__main__":
    app()
