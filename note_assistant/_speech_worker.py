"""Apple Speech recognition subprocess worker.

Runs as `python -m note_assistant._speech_worker`.  Reads audio chunks from
stdin, transcribes via SFSpeechRecognizer, writes text to stdout.  Running in
a dedicated subprocess avoids the asyncio/CFRunLoop incompatibility that
prevents Speech callbacks from being delivered inside Textual's event loop.

Protocol (all lengths big-endian uint32):
  Request:  [audio_len 4B][audio float32 bytes][sample_rate 4B]
  Response: [text_len  4B][utf-8 text bytes]
"""
from __future__ import annotations

import struct
import sys
import threading
import time

import numpy as np


def _transcribe(audio: np.ndarray, sample_rate: int) -> str:
    import Speech
    import AVFoundation
    import Foundation
    from CoreFoundation import CFRunLoopRunInMode, kCFRunLoopDefaultMode

    if float(np.sqrt(np.mean(audio ** 2))) < 1e-4:
        return ""

    fmt = AVFoundation.AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
        AVFoundation.AVAudioPCMFormatFloat32, sample_rate, 1, True
    )
    buf = AVFoundation.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(fmt, len(audio))
    buf.setFrameLength_(len(audio))
    src = audio.tobytes()
    ch = buf.floatChannelData()[0]
    memoryview(ch.as_buffer(len(src))).cast("B")[: len(src)] = memoryview(src).cast("B")

    recognizer = Speech.SFSpeechRecognizer.alloc().initWithLocale_(
        Foundation.NSLocale.alloc().initWithLocaleIdentifier_("en-US")
    )
    if not recognizer or not recognizer.isAvailable():
        return ""

    recognizer.setSupportsOnDeviceRecognition_(True)
    request = Speech.SFSpeechAudioBufferRecognitionRequest.alloc().init()
    request.appendAudioPCMBuffer_(buf)
    request.endAudio()

    result_text = ""
    done = threading.Event()

    def handler(result, error):  # noqa: ANN001
        nonlocal result_text
        if result:
            result_text = result.bestTranscription().formattedString()
        if (result and result.isFinal()) or not result:
            done.set()

    recognizer.recognitionTaskWithRequest_resultHandler_(request, handler)
    deadline = time.monotonic() + 10.0
    while not done.is_set() and time.monotonic() < deadline:
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.1, True)

    return result_text.strip()


def main() -> None:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        header = stdin.read(8)
        if len(header) < 8:
            break
        audio_len, sample_rate = struct.unpack(">II", header)
        audio_bytes = stdin.read(audio_len)
        if len(audio_bytes) < audio_len:
            break

        audio = np.frombuffer(audio_bytes, dtype=np.float32).copy()
        try:
            text = _transcribe(audio, sample_rate)
        except Exception:
            text = ""

        encoded = text.encode("utf-8")
        stdout.write(struct.pack(">I", len(encoded)) + encoded)
        stdout.flush()


if __name__ == "__main__":
    main()
