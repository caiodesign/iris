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
