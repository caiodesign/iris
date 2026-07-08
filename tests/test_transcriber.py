# tests/test_transcriber.py
import io
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from companion.transcriber import (
    LocalTranscriber,
    OpenAITranscriber,
    _encode_wav,
    make_transcriber,
)


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


def test_encode_wav_is_16k_mono_pcm16_and_round_trips():
    samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)

    data = _encode_wav(samples, sample_rate=16000)

    assert data[:4] == b"RIFF"
    with wave.open(io.BytesIO(data), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        frames = wav.readframes(wav.getnframes())

    decoded = np.frombuffer(frames, dtype="<i2")
    expected = np.array([0, 16383, -16383, 32767, -32767], dtype="<i2")
    assert np.array_equal(decoded, expected)


def test_encode_wav_clips_out_of_range_samples():
    data = _encode_wav(np.array([2.0, -2.0], dtype=np.float32))
    with wave.open(io.BytesIO(data), "rb") as wav:
        decoded = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert np.array_equal(decoded, np.array([32767, -32767], dtype="<i2"))


def test_openai_transcriber_sends_wav_and_strips_reply():
    fake_result = MagicMock()
    fake_result.text = "  Hello there  "
    with patch("companion.transcriber.OpenAI") as MockOpenAI:
        client = MockOpenAI.return_value
        client.audio.transcriptions.create.return_value = fake_result

        transcriber = OpenAITranscriber("gpt-4o-transcribe")
        result = transcriber.transcribe(np.zeros(16000, dtype=np.float32))

    assert result == "Hello there"
    kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-transcribe"
    filename, data, mime = kwargs["file"]
    assert filename == "speech.wav"
    assert mime == "audio/wav"
    assert data[:4] == b"RIFF"


def test_make_transcriber_builds_openai():
    with patch("companion.transcriber.OpenAI"):
        transcriber = make_transcriber("openai")
    assert isinstance(transcriber, OpenAITranscriber)
