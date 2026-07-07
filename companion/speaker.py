# companion/speaker.py
import io
import wave

import numpy as np
import sounddevice as sd
from piper import PiperVoice


class Speaker:
    def __init__(self, voice_path: str):
        self.voice = PiperVoice.load(voice_path)

    def speak(self, text: str) -> None:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file)

        buffer.seek(0)
        with wave.open(buffer, "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_rate = wav_file.getframerate()

        audio = np.frombuffer(frames, dtype=np.int16)
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
