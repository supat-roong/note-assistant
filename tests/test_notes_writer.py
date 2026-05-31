import time
import pytest
from pathlib import Path
from note_assistant.notes_writer import NotesWriter


@pytest.fixture
def writer(monkeypatch):
    """NotesWriter with osascript patched out."""
    calls = []
    monkeypatch.setattr(NotesWriter, "_run_osascript", lambda self, s: calls.append(s) or "")
    nw = NotesWriter("Test — {date}")
    nw._note_id = "fake-id"
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
    assert "Session ended" in calls[0]


def test_escape_double_quotes():
    assert NotesWriter._as('say "hi"') == 'say \\"hi\\"'


def test_escape_newline():
    assert NotesWriter._as("line\nbreak") == "line\\nbreak"


def test_escape_backslash():
    assert NotesWriter._as("a\\b") == "a\\\\b"


def test_summary_to_html_no_p_or_heading_elements():
    """Apple Notes overlaps content when <p> or heading elements are present."""
    html = NotesWriter._summary_to_html("- Point one\n- Point two\n\nPlain paragraph.")
    for tag in ("<p>", "</p>", "<h1>", "<h2>", "<h3>"):
        assert tag not in html, f"Block element {tag!r} must not appear in Apple Notes HTML"
    assert "Point one" in html
    assert "Point two" in html
    assert "Plain paragraph" in html
    assert "<li>" in html


def test_write_title_only_resets_body_to_title_only(writer):
    nw, calls = writer
    nw._summary = "some summary"
    nw._transcript_lines = ["some text"]
    nw.write_title_only()
    assert len(calls) == 1
    assert "some summary" not in calls[0]
    assert "some text" not in calls[0]
    assert nw._title in calls[0] or "Test" in calls[0]


def test_write_title_only_skips_when_note_not_created(monkeypatch):
    calls = []
    monkeypatch.setattr(NotesWriter, "_run_osascript", lambda self, s: calls.append(s) or "")
    nw = NotesWriter("Test")
    nw._note_created = False
    nw.write_title_only()
    assert calls == []


def test_finalize_session_forces_flush_with_full_content(writer):
    nw, calls = writer
    nw._summary = "bullet point"
    nw._transcript_lines = ["some text"]
    nw._last_flush = time.monotonic()  # throttled
    nw.finalize_session()
    assert len(calls) == 1
    assert "bullet point" in calls[0]
    assert "some text" in calls[0]
    assert "Session ended" in calls[0]


def test_attach_recording_calls_osascript_with_path(writer, tmp_path):
    nw, calls = writer
    fake_m4a = tmp_path / "recording.m4a"
    fake_m4a.touch()
    nw.attach_recording(fake_m4a)
    assert len(calls) == 1
    assert "make new attachment" in calls[0]
    assert str(fake_m4a) in calls[0]


def test_attach_recording_skips_when_note_not_created(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(NotesWriter, "_run_osascript", lambda self, s: calls.append(s) or "")
    nw = NotesWriter("Test")
    nw._note_created = False
    nw.attach_recording(tmp_path / "recording.m4a")
    assert calls == []
