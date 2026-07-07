# tests/test_transcriber.py
from unittest.mock import MagicMock, patch

import numpy as np

from companion.transcriber import Transcriber


def test_transcribe_joins_and_strips_segment_text():
    fake_segment_1 = MagicMock(text=" Hello ")
    fake_segment_2 = MagicMock(text=" world ")

    with patch("companion.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (
            [fake_segment_1, fake_segment_2],
            None,
        )
        transcriber = Transcriber("base.en", "cpu", "int8")
        result = transcriber.transcribe(np.zeros(16000, dtype=np.float32))

    assert result == "Hello world"
    MockModel.assert_called_once_with("base.en", device="cpu", compute_type="int8")


def test_transcribe_returns_empty_string_for_silence():
    with patch("companion.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = ([], None)
        transcriber = Transcriber("base.en", "cpu", "int8")
        result = transcriber.transcribe(np.zeros(16000, dtype=np.float32))

    assert result == ""
