import time
import pytest
from note_assistant.notes_writer import NotesWriter


@pytest.fixture
def writer(monkeypatch):
    """NotesWriter with osascript patched out."""
    calls = []
    monkeypatch.setattr(NotesWriter, "_run_osascript", lambda self, s: calls.append(s))
    nw = NotesWriter("Test — {date}")
    nw._note_created = True
    nw._last_flush = 0.0  # throttle expired
    return nw, calls


def test_transcript_lines_accumulate(writer):
    nw, _ = writer
    nw.append_transcript("hello")
    nw.append_transcript("world")
    assert nw._transcript_lines == ["hello", "world"]


def test_summary_is_replaced(writer):
    nw, _ = writer
    nw.update_summary("first")
    nw.update_summary("second")
    assert nw._summary == "second"


def test_flush_writes_body_containing_transcript(writer):
    nw, calls = writer
    nw.append_transcript("spoken text")
    assert len(calls) == 1
    assert "spoken text" in calls[0]


def test_flush_throttled(writer, monkeypatch):
    nw, calls = writer
    nw._last_flush = time.monotonic()  # pretend flush just happened
    nw.append_transcript("should not flush")
    assert len(calls) == 0


def test_flush_includes_summary_when_present(writer):
    nw, calls = writer
    nw._summary = "bullet point"
    nw.append_transcript("text")
    assert "bullet point" in calls[0]


def test_close_session_forces_flush_despite_throttle(writer):
    nw, calls = writer
    nw._last_flush = time.monotonic()  # throttled
    nw.close_session()
    assert len(calls) == 1
    assert "_Session ended._" in calls[0]


def test_escape_double_quotes():
    assert NotesWriter._escape('say "hi"') == 'say \\"hi\\"'


def test_escape_newline():
    assert NotesWriter._escape("line\nbreak") == "line\\nbreak"


def test_escape_backslash():
    assert NotesWriter._escape("a\\b") == "a\\\\b"
