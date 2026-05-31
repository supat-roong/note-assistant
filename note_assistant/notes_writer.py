"""Apple Notes writer — batched in-memory writes via JXA/osascript."""
from __future__ import annotations

import subprocess
import time
from datetime import datetime
from pathlib import Path


class NotesWriter:
    """Writes transcript + summary live into an Apple Notes note.

    Maintains in-memory buffers and flushes once per throttle window.
    Body is sent as HTML so Apple Notes renders line breaks correctly.
    """

    def __init__(self, title_template: str = "Note Assistant — {date}"):
        self._date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._title = title_template.format(date=self._date_str)
        self._last_flush = 0.0
        self._throttle_secs = 1.0
        self._note_id: str = ""
        self._note_created = False
        self._transcript_lines: list[str] = []
        self._summary: str = ""
        self._closed = False
        self._recording_name: str = ""

    def open_session(self) -> None:
        self._note_id = self._create_note(self._build_html())
        self._note_created = bool(self._note_id)

    def append_transcript(self, text: str) -> None:
        self._transcript_lines.append(text)
        self._maybe_flush()

    def update_summary(self, summary: str) -> None:
        self._summary = summary
        self._maybe_flush()

    def set_title(self, new_title: str) -> None:
        """Rename the note with an informative title generated after processing."""
        self._title = new_title
        self._flush(force=True)

    def set_recording(self, path: Path) -> None:
        """Store recording filename to display in note body between title and summary."""
        self._recording_name = path.name

    def close_session(self) -> None:
        self.finalize_session()

    def write_title_only(self) -> None:
        """Reset note body to title heading only, before attaching a recording."""
        if not self._note_created:
            return
        html = f'<div><b>{self._he(self._title)}</b></div><div><br></div>'
        script = f"""
        tell application "Notes"
            set targetNote to note id "{self._note_id}"
            set body of targetNote to "{self._as(html)}"
        end tell
        """
        self._run_osascript(script)

    def finalize_session(self) -> None:
        """Write full note content (summary + transcript) and mark session ended."""
        self._closed = True
        self._flush(force=True)

    def attach_recording(self, path: Path) -> bool:
        """Attach audio file to note via Edit > Attach File dialog (requires Accessibility).

        Must be called AFTER the final body write — any subsequent set body will
        destroy the attachment. Returns True if attachment was attempted.
        """
        if not self._note_created:
            return False
        abs_path = path.resolve()
        script = f"""
        tell application "Notes"
            activate
            show note id "{self._note_id}"
        end tell
        delay 0.5
        -- Click note body to focus it for editing
        tell application "System Events"
            tell process "Notes"
                set w to window 1
                set winPos to position of w
                set winSize to size of w
                click at {{(item 1 of winPos) + round((item 1 of winSize) * 0.65), (item 2 of winPos) + round((item 2 of winSize) * 0.4)}}
            end tell
        end tell
        delay 0.3
        -- Go to end of note so attachment lands after all content
        tell application "System Events"
            key code 119 using command down
        end tell
        delay 0.2
        -- Open Attach File dialog
        tell application "System Events"
            tell process "Notes"
                click menu item "Attach File\\u2026" of menu "Edit" of menu bar 1
            end tell
        end tell
        delay 1.0
        -- Navigate to file via Go to Folder (Cmd+Shift+G)
        tell application "System Events"
            keystroke "g" using {{command down, shift down}}
            delay 0.5
            keystroke "{self._as(str(abs_path))}"
            delay 0.3
            keystroke return
            delay 0.5
            keystroke return
        end tell
        delay 0.5
        """
        result = self._run_osascript(script)
        return True

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
        if self._recording_name:
            parts.append(f"<div>🎙 {self._he(self._recording_name)}</div>")
            parts.append("<div><br></div>")
        if self._summary:
            parts.append("<div><b>Summary</b></div>")
            parts.append(self._summary_to_html(self._summary))
            parts.append("<div><br></div>")
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
    def _summary_to_html(text: str) -> str:
        """Convert Markdown summary to Apple Notes-compatible HTML.

        Apple Notes handles <ul>/<li> fine but renders <p> and headings at
        incorrect positions alongside <div> content, causing visible overlap.
        """
        import re
        import markdown

        raw = markdown.markdown(text, extensions=["nl2br"])

        # Unwrap <p> inside loose list items before converting standalone <p>.
        raw = re.sub(
            r"<li>\s*<p>(.*?)</p>\s*</li>",
            lambda m: f"<li>{m.group(1).strip()}</li>",
            raw,
            flags=re.DOTALL,
        )
        raw = re.sub(
            r"<h\d>(.*?)</h\d>",
            lambda m: f"<div><b>{m.group(1)}</b></div>",
            raw,
            flags=re.DOTALL,
        )
        raw = re.sub(
            r"<p>(.*?)</p>",
            lambda m: f"<div>{m.group(1)}</div>",
            raw,
            flags=re.DOTALL,
        )
        # Strip bare newlines so _as() doesn't embed literal \n in the AppleScript string.
        return raw.replace("\n", "")

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
