# tests/test_transcriber.py
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from companion.transcriber import LocalTranscriber, make_transcriber


def test_transcribe_joins_and_strips_segment_text():
    fake_segment_1 = MagicMock(text=" Hello ")
    fake_segment_2 = MagicMock(text=" world ")

    with patch("companion.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (
            [fake_segment_1, fake_segment_2],
            None,
        )
        transcriber = LocalTranscriber("base.en", "cpu", "int8")
        result = transcriber.transcribe(np.zeros(16000, dtype=np.float32))

    assert result == "Hello world"
    MockModel.assert_called_once_with("base.en", device="cpu", compute_type="int8")


def test_transcribe_returns_empty_string_for_silence():
    with patch("companion.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = ([], None)
        transcriber = LocalTranscriber("base.en", "cpu", "int8")
        result = transcriber.transcribe(np.zeros(16000, dtype=np.float32))

    assert result == ""


def test_init_registers_pip_nvidia_dll_dirs_before_loading_model():
    # pip's nvidia-* wheels put cuBLAS/cuDNN DLLs in site-packages/nvidia/*/bin,
    # which is not on Windows' DLL search path; without registration CTranslate2
    # crashes with "cublas64_12.dll is not found" at the first transcription.
    with patch("companion.transcriber.WhisperModel"), patch(
        "companion.transcriber.site.getsitepackages", return_value=["fake_site"]
    ), patch(
        "companion.transcriber.site.getusersitepackages", return_value="fake_user_site"
    ), patch(
        "companion.transcriber.glob.glob", return_value=["fake_site/nvidia/cublas/bin"]
    ), patch(
        "companion.transcriber.os.add_dll_directory", create=True
    ) as mock_add_dll:
        LocalTranscriber("base.en", "cuda", "float16")

    mock_add_dll.assert_any_call("fake_site/nvidia/cublas/bin")


def test_make_transcriber_builds_local():
    with patch("companion.transcriber.WhisperModel"):
        transcriber = make_transcriber("local")
    assert isinstance(transcriber, LocalTranscriber)


def test_make_transcriber_rejects_unknown_name():
    with pytest.raises(ValueError):
        make_transcriber("robot-ears")
