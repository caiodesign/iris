# tests/test_speaker.py
from unittest.mock import patch

import numpy as np

from companion.speaker import Speaker


def fake_synthesize_wav(text, wav_file):
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(22050)
    wav_file.writeframes(np.array([100, -100, 200], dtype=np.int16).tobytes())


def test_speak_plays_synthesized_audio_at_voice_sample_rate():
    with patch("companion.speaker.PiperVoice") as MockVoice, patch(
        "companion.speaker.sd"
    ) as mock_sd:
        MockVoice.load.return_value.synthesize_wav.side_effect = fake_synthesize_wav

        speaker = Speaker("fake_voice.onnx")
        speaker.speak("Hello there")

    played_audio_arg = mock_sd.play.call_args.args[0]
    np.testing.assert_array_equal(
        played_audio_arg, np.array([100, -100, 200], dtype=np.int16)
    )
    assert mock_sd.play.call_args.kwargs["samplerate"] == 22050
    mock_sd.wait.assert_called_once()
