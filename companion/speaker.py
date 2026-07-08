# companion/speaker.py
import sounddevice as sd
from kokoro_onnx import Kokoro


class Speaker:
    def __init__(self, model_path: str, voices_path: str, voice: str, speed: float):
        self.kokoro = Kokoro(model_path, voices_path)
        self.voice = voice
        self.speed = speed

    def speak(self, text: str) -> None:
        samples, sample_rate = self.kokoro.create(
            text, voice=self.voice, speed=self.speed, lang="en-us"
        )
        sd.play(samples, samplerate=sample_rate)
        sd.wait()
