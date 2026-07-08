# companion/voice_detector.py
import collections

import numpy as np
import sounddevice as sd
import webrtcvad


class VoiceDetector:
    def __init__(
        self,
        sample_rate: int,
        frame_duration_ms: int,
        silence_timeout_ms: int,
        preroll_ms: int,
        vad_aggressiveness: int,
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.silence_timeout_frames = silence_timeout_ms // frame_duration_ms
        self.preroll_frames = preroll_ms // frame_duration_ms
        self.vad = webrtcvad.Vad(vad_aggressiveness)

    def listen_for_utterance(self, stop_check=None) -> "np.ndarray | None":
        frames = []
        # Ring buffer of the most recent pre-speech frames; prepended on
        # trigger so the first syllable isn't clipped off the utterance.
        preroll = collections.deque(maxlen=self.preroll_frames)
        triggered = False
        silence_count = 0

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_size,
        ) as stream:
            while True:
                # Checked every frame (~30 ms) so the web UI's End button
                # interrupts promptly even while waiting for speech.
                if stop_check is not None and stop_check():
                    return None
                frame, _ = stream.read(self.frame_size)
                is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)

                if not triggered:
                    if is_speech:
                        frames.extend(preroll)
                        frames.append(frame)
                        triggered = True
                    else:
                        preroll.append(frame)
                else:
                    frames.append(frame)
                    if is_speech:
                        silence_count = 0
                    else:
                        silence_count += 1
                        if silence_count > self.silence_timeout_frames:
                            break

        audio = np.concatenate(frames, axis=0).flatten().astype(np.float32) / 32768.0
        return audio
