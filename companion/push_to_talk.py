# companion/push_to_talk.py
from pynput import keyboard, mouse

import threading

import numpy as np
import sounddevice as sd

# Mouse side buttons are platform-specific members of pynput's Button enum:
# Linux X11 exposes button8/button9; Windows exposes x1/x2. Try each in order
# and use the first that exists on this platform.
_MOUSE_ALIASES = {
    "MOUSE_LEFT": ("left",),
    "MOUSE_RIGHT": ("right",),
    "MOUSE_MIDDLE": ("middle",),
    "MOUSE_4": ("button8", "x1"),
    "MOUSE_5": ("button9", "x2"),
}


def _unbindable(key_str):
    return ValueError(
        f"Could not bind PTT key '{key_str}' on this platform. Set PTT_KEY in "
        'companion/config.py to a keyboard key such as "space".'
    )


def resolve_trigger(key_str):
    """Map a PTT_KEY string to a concrete pynput target.

    Returns ("mouse", Button) or ("keyboard", Key|KeyCode). Raises ValueError
    if the name cannot be bound on this platform."""
    if key_str in _MOUSE_ALIASES:
        for attr in _MOUSE_ALIASES[key_str]:
            button = getattr(mouse.Button, attr, None)
            if button is not None:
                return ("mouse", button)
        raise _unbindable(key_str)

    named = getattr(keyboard.Key, key_str, None)
    if named is not None:
        return ("keyboard", named)
    if len(key_str) == 1:
        return ("keyboard", keyboard.KeyCode.from_char(key_str))
    raise _unbindable(key_str)


class PushToTalkRecorder:
    """Hold-to-talk capture. A background pynput listener sets/clears a
    threading.Event while the configured button is held; listen_for_utterance
    records mic frames for exactly that window and returns the same float32
    array shape VoiceDetector produces, so the rest of the pipeline is
    unchanged."""

    def __init__(self, sample_rate: int, frame_duration_ms: int, ptt_key: str):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self._pressed = threading.Event()

        kind, target = resolve_trigger(ptt_key)
        self._target = target
        if kind == "mouse":
            self._listener = mouse.Listener(on_click=self._on_click)
        else:
            self._listener = keyboard.Listener(
                on_press=self._on_press, on_release=self._on_release
            )
        self._listener.daemon = True
        self._listener.start()

    def _on_click(self, x, y, button, pressed):
        if button == self._target:
            if pressed:
                self._pressed.set()
            else:
                self._pressed.clear()

    def _on_press(self, key):
        if key == self._target:
            self._pressed.set()

    def _on_release(self, key):
        if key == self._target:
            self._pressed.clear()

    def listen_for_utterance(self, stop_check=None) -> "np.ndarray | None":
        # Wait for the button in short slices instead of a single blocking
        # wait() so a stop request interrupts within ~100 ms.
        while not self._pressed.wait(timeout=0.1):
            if stop_check is not None and stop_check():
                return None
        frames = []
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_size,
        ) as stream:
            while self._pressed.is_set():
                if stop_check is not None and stop_check():
                    return None
                frame, _ = stream.read(self.frame_size)
                frames.append(frame)

        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames, axis=0).flatten().astype(np.float32) / 32768.0

    def close(self) -> None:
        self._listener.stop()
