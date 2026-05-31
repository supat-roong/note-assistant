import subprocess
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from unittest.mock import patch, MagicMock

from note_assistant.recorder import SessionRecorder


def test_start_creates_wav_file(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec.start()
    rec.write(np.zeros(1600, dtype=np.float32))
    rec._sf.close()
    assert rec._wav_path.exists()


def test_write_appends_chunks(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec.start()
    rec.write(np.ones(1600, dtype=np.float32))
    rec.write(np.ones(1600, dtype=np.float32))
    rec._sf.close()
    data, sr = sf.read(str(rec._wav_path))
    assert len(data) == 3200
    assert sr == 16000


def test_finish_calls_ffmpeg_for_mp3(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec.start()
    rec.write(np.zeros(160, dtype=np.float32))

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        mp3 = rec.finish()

    assert mock_run.call_count == 1
    cmd = mock_run.call_args_list[0][0][0]
    assert ".mp3" in str(cmd)


def test_finish_returns_mp3_path(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec.start()
    rec.write(np.zeros(160, dtype=np.float32))

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        mp3 = rec.finish()

    assert mp3.suffix == ".mp3"
    assert mp3.parent == tmp_path


def test_finish_raises_when_ffmpeg_missing(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec.start()
    rec.write(np.zeros(160, dtype=np.float32))

    with patch("subprocess.run", side_effect=FileNotFoundError("ffmpeg not found")):
        with pytest.raises(FileNotFoundError):
            rec.finish()


def test_finish_raises_on_ffmpeg_nonzero_exit(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec.start()
    rec.write(np.zeros(160, dtype=np.float32))

    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg", stderr=b"error")):
        with pytest.raises(subprocess.CalledProcessError):
            rec.finish()


def test_cleanup_removes_wav(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec._wav_path.touch()
    rec.cleanup()
    assert not rec._wav_path.exists()


def test_cleanup_does_not_remove_mp3(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec._wav_path.touch()
    rec._mp3_path.touch()
    rec.cleanup()
    assert rec._mp3_path.exists()


def test_cleanup_is_safe_when_files_missing(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec.cleanup()  # no files exist — must not raise


def test_finish_raises_if_not_started(tmp_path):
    rec = SessionRecorder(tmp_path)
    with pytest.raises(RuntimeError, match="before start"):
        rec.finish()


def test_finish_raises_if_called_twice(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec.start()
    rec.write(np.zeros(160, dtype=np.float32))
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        rec.finish()
    with pytest.raises(RuntimeError, match="already called"):
        rec.finish()
