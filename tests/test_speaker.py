# tests/test_speaker.py
from unittest.mock import patch

import numpy as np

from companion.speaker import Speaker


def test_speak_plays_kokoro_audio_at_returned_sample_rate():
    fake_samples = np.array([0.1, -0.2, 0.3], dtype=np.float32)

    with patch("companion.speaker.Kokoro") as MockKokoro, patch(
        "companion.speaker.sd"
    ) as mock_sd:
        MockKokoro.return_value.create.return_value = (fake_samples, 24000)

        speaker = Speaker("fake_model.onnx", "fake_voices.bin", "am_michael", 1.0)
        speaker.speak("Hello there")

    MockKokoro.assert_called_once_with("fake_model.onnx", "fake_voices.bin")
    MockKokoro.return_value.create.assert_called_once_with(
        "Hello there", voice="am_michael", speed=1.0, lang="en-us"
    )
    played_audio_arg = mock_sd.play.call_args.args[0]
    np.testing.assert_array_equal(played_audio_arg, fake_samples)
    assert mock_sd.play.call_args.kwargs["samplerate"] == 24000
    mock_sd.wait.assert_called_once()
