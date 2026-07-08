# companion/push_to_talk.py
from pynput import keyboard, mouse

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
