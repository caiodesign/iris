# tests/test_push_to_talk.py
import types

import pytest

from companion.push_to_talk import resolve_trigger


def test_resolve_trigger_maps_a_standard_mouse_button():
    kind, target = resolve_trigger("MOUSE_LEFT")
    from pynput import mouse

    assert kind == "mouse"
    assert target == mouse.Button.left


def test_resolve_trigger_maps_a_named_keyboard_key():
    kind, target = resolve_trigger("space")
    from pynput import keyboard

    assert kind == "keyboard"
    assert target == keyboard.Key.space


def test_resolve_trigger_maps_a_single_character_key():
    kind, target = resolve_trigger("a")
    from pynput import keyboard

    assert kind == "keyboard"
    assert target == keyboard.KeyCode.from_char("a")


def test_resolve_trigger_rejects_an_unknown_name():
    with pytest.raises(ValueError):
        resolve_trigger("not_a_real_key")


def test_resolve_trigger_raises_when_side_button_missing_on_platform():
    # Simulate a platform whose Button enum lacks the side buttons: getattr
    # then falls through both candidate attribute names to the ValueError.
    fake_button = types.SimpleNamespace(left="L", right="R", middle="M")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("companion.push_to_talk.mouse.Button", fake_button)
        with pytest.raises(ValueError):
            resolve_trigger("MOUSE_4")


from unittest.mock import patch

import numpy as np

from companion.push_to_talk import PushToTalkRecorder


def _build_recorder():
    # Patch both Listener classes so constructing the recorder never starts a
    # real global hook; resolve_trigger("MOUSE_LEFT") still uses the real enum.
    with patch("companion.push_to_talk.mouse.Listener"), patch(
        "companion.push_to_talk.keyboard.Listener"
    ):
        return PushToTalkRecorder(16000, 10, "MOUSE_LEFT")


def test_recorder_captures_frames_while_button_held():
    recorder = _build_recorder()
    frame = (np.ones((160, 1), dtype=np.int16) * 100)

    reads = {"n": 0}

    def fake_read(size):
        reads["n"] += 1
        if reads["n"] >= 3:  # release after the third frame
            recorder._pressed.clear()
        return frame, False

    with patch("companion.push_to_talk.sd.InputStream") as MockStream:
        MockStream.return_value.__enter__.return_value.read.side_effect = fake_read
        recorder._pressed.set()
        audio = recorder.listen_for_utterance()

    assert audio.dtype == np.float32
    assert len(audio) == 3 * 160
    assert np.allclose(audio, 100 / 32768.0)


def test_recorder_returns_empty_when_released_immediately():
    recorder = _build_recorder()

    with patch("companion.push_to_talk.sd.InputStream") as MockStream:
        stream = MockStream.return_value.__enter__.return_value
        # The button comes up exactly as the stream opens: no frame is read.
        MockStream.return_value.__enter__.side_effect = (
            lambda: recorder._pressed.clear() or stream
        )
        recorder._pressed.set()
        audio = recorder.listen_for_utterance()

    assert audio.dtype == np.float32
    assert len(audio) == 0
    stream.read.assert_not_called()


def test_click_callback_tracks_only_the_target_button():
    from pynput import mouse

    recorder = _build_recorder()  # target is Button.left

    recorder._on_click(0, 0, mouse.Button.left, True)
    assert recorder._pressed.is_set()
    recorder._on_click(0, 0, mouse.Button.left, False)
    assert not recorder._pressed.is_set()

    # A different button must not touch the flag.
    recorder._pressed.set()
    recorder._on_click(0, 0, mouse.Button.right, False)
    assert recorder._pressed.is_set()


def test_close_stops_the_listener():
    with patch("companion.push_to_talk.mouse.Listener") as MockListener, patch(
        "companion.push_to_talk.keyboard.Listener"
    ):
        recorder = PushToTalkRecorder(16000, 10, "MOUSE_LEFT")
        recorder.close()

    MockListener.return_value.stop.assert_called_once()


def test_ptt_listen_returns_none_when_stop_requested_before_press():
    recorder = _build_recorder()
    # Button never pressed: without stop_check this would block forever.
    audio = recorder.listen_for_utterance(stop_check=lambda: True)
    assert audio is None


def test_ptt_listen_returns_none_when_stop_requested_mid_recording():
    recorder = _build_recorder()
    frame = np.ones((160, 1), dtype=np.int16)
    calls = {"n": 0}

    def stop_check():
        calls["n"] += 1
        return calls["n"] >= 3

    with patch("companion.push_to_talk.sd.InputStream") as MockStream:
        MockStream.return_value.__enter__.return_value.read.return_value = (
            frame,
            False,
        )
        recorder._pressed.set()
        audio = recorder.listen_for_utterance(stop_check=stop_check)

    assert audio is None
