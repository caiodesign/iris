# tests/test_voice_detector.py
import types
from unittest.mock import patch

import numpy as np

from companion.voice_detector import VoiceDetector


def _silent_detector():
    detector = VoiceDetector(
        sample_rate=16000,
        frame_duration_ms=30,
        silence_timeout_ms=2000,
        preroll_ms=300,
        vad_aggressiveness=2,
    )
    # webrtcvad.Vad is a C extension; swap the whole attribute instead of
    # patching a method on it.
    detector.vad = types.SimpleNamespace(is_speech=lambda data, rate: False)
    return detector


def test_listen_returns_none_when_stop_requested():
    detector = _silent_detector()
    frame = np.zeros((480, 1), dtype=np.int16)
    calls = {"n": 0}

    def stop_check():
        calls["n"] += 1
        return calls["n"] >= 3

    with patch("companion.voice_detector.sd.InputStream") as MockStream:
        MockStream.return_value.__enter__.return_value.read.return_value = (
            frame,
            False,
        )
        result = detector.listen_for_utterance(stop_check=stop_check)

    assert result is None


def test_listen_without_stop_check_still_captures_speech():
    detector = _silent_detector()
    frame = np.ones((480, 1), dtype=np.int16) * 100
    # Speech for 2 frames, then silence until the timeout trips.
    speech = {"n": 0}

    def is_speech(data, rate):
        speech["n"] += 1
        return speech["n"] <= 2

    detector.vad = types.SimpleNamespace(is_speech=is_speech)
    with patch("companion.voice_detector.sd.InputStream") as MockStream:
        MockStream.return_value.__enter__.return_value.read.return_value = (
            frame,
            False,
        )
        audio = detector.listen_for_utterance()

    assert audio is not None
    assert audio.dtype == np.float32
    assert len(audio) > 0
