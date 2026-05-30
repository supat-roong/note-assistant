"""Apple Notes writer — batched in-memory writes via JXA/osascript."""
from __future__ import annotations

import subprocess
import time
from datetime import datetime


class NotesWriter:
    """Writes transcript + summary live into an Apple Notes note.

    Maintains in-memory buffers and flushes once per throttle window.
    Body is sent as HTML so Apple Notes renders line breaks correctly.
    """

    def __init__(self, title_template: str = "Note Assistant — {date}"):
        self._title = title_template.format(
            date=datetime.now().strftime("%Y-%m-%d %H:%M")
        )
        self._last_flush = 0.0
        self._throttle_secs = 1.0
        self._note_id: str = ""
        self._note_created = False
        self._transcript_lines: list[str] = []
        self._summary: str = ""
        self._closed = False

    def open_session(self) -> None:
        self._note_id = self._create_note(self._build_html())
        self._note_created = bool(self._note_id)

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
        self._flush()

    def _flush(self, force: bool = False) -> None:
        if not self._note_created:
            return
        now = time.monotonic()
        if not force and now - self._last_flush < self._throttle_secs:
            return
        self._last_flush = now
        script = f"""
        tell application "Notes"
            set targetNote to note id "{self._note_id}"
            set body of targetNote to "{self._as(self._build_html())}"
        end tell
        """
        self._run_osascript(script)

    def _build_html(self) -> str:
        """Build the full note body as HTML (Apple Notes uses HTML internally)."""
        parts = [
            f"<div><b>{self._he(self._title)}</b></div>",
            "<div><br></div>",
        ]
        if self._summary:
            parts += [
                "<div><b>Summary</b></div>",
                f"<div>{self._he(self._summary)}</div>",
                "<div><br></div>",
            ]
        parts.append("<div><b>Transcript</b></div>")
        for line in self._transcript_lines:
            parts.append(f"<div>{self._he(line)}</div>")
        if self._closed:
            parts += ["<div><br></div>", "<div><i>Session ended.</i></div>"]
        return "".join(parts)

    def _create_note(self, html: str) -> str:
        script = f"""
        tell application "Notes"
            set newNote to make new note at folder "Notes" with properties {{name:"{self._title}", body:"{self._as(html)}"}}
            return id of newNote
        end tell
        """
        return self._run_osascript(script)

    @staticmethod
    def _run_osascript(script: str) -> str:
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""

    @staticmethod
    def _he(text: str) -> str:
        """Escape text for safe embedding in HTML."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    @staticmethod
    def _as(text: str) -> str:
        """Escape HTML string for safe embedding in an AppleScript string literal."""
        return (
            text
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "")
        )
