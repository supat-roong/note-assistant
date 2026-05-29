"""Apple Notes writer — batched in-memory writes via JXA/osascript."""
from __future__ import annotations

import subprocess
import time
from datetime import datetime


class NotesWriter:
    """Writes transcript + summary live into an Apple Notes note.

    Maintains in-memory buffers and flushes once per throttle window
    instead of reading + rewriting on every append call.
    """

    def __init__(self, title_template: str = "Note Assistant — {date}"):
        self._title = title_template.format(
            date=datetime.now().strftime("%Y-%m-%d %H:%M")
        )
        self._last_flush = 0.0
        self._throttle_secs = 1.0
        self._note_created = False
        self._transcript_lines: list[str] = []
        self._summary: str = ""
        self._closed = False

    def open_session(self) -> None:
        header = f"# {self._title}\n\n---\n\n## Transcript\n\n"
        self._create_note(header)
        self._note_created = True

    def append_transcript(self, text: str) -> None:
        self._transcript_lines.append(text)
        self._maybe_flush()

    def update_summary(self, summary: str) -> None:
        self._summary = summary
        self._maybe_flush()

    def close_session(self) -> None:
        self._closed = True
        self._flush(force=True)

    def _maybe_flush(self) -> None:
        now = time.monotonic()
        if now - self._last_flush >= self._throttle_secs:
            self._flush()

    def _flush(self, force: bool = False) -> None:
        if not self._note_created:
            return
        now = time.monotonic()
        if not force and now - self._last_flush < self._throttle_secs:
            return
        self._last_flush = now

        body = f"# {self._title}\n\n---\n\n## Transcript\n\n"
        body += " ".join(self._transcript_lines)
        if self._summary:
            body += f"\n\n---\n\n## Summary\n\n{self._summary}"
        if self._closed:
            body += "\n\n---\n_Session ended._"

        script = f"""
        tell application "Notes"
            set targetNote to first note whose name is "{self._title}"
            set body of targetNote to "{self._escape(body)}"
        end tell
        """
        self._run_osascript(script)

    def _create_note(self, body: str) -> None:
        script = f"""
        tell application "Notes"
            make new note at folder "Notes" with properties {{name:"{self._title}", body:"{self._escape(body)}"}}
        end tell
        """
        self._run_osascript(script)

    @staticmethod
    def _run_osascript(script: str) -> None:
        try:
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    @staticmethod
    def _escape(text: str) -> str:
        return (
            text
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "")
        )
