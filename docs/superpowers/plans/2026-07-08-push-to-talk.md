# Push-to-Talk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional push-to-talk capture mode: hold a configured button (default `MOUSE_4`) to record instead of voice-activity detection, selectable at launch.

**Architecture:** PTT is a pure capture swap. A new `PushToTalkRecorder` exposes the same `listen_for_utterance() -> np.ndarray` interface as `VoiceDetector`, so the main loop, state machine, wake word, and memory are unchanged. A `pynput` global listener toggles a `threading.Event` while the button is held; `main.py` picks the recorder or the detector at startup.

**Tech Stack:** Python, `pynput` (new — global input hook), `sounddevice`, `numpy`, pytest + `unittest.mock`.

## Global Constraints

- **One new dependency only:** `pynput>=1.7,<2`. No others.
- **`pynput` is imported lazily in `main.py`** — only inside the `if ptt:` branch — so VAD-only users never load it. The `companion/push_to_talk.py` module itself imports `pynput` at top (it is only imported when PTT is enabled).
- **Identical audio contract:** the recorder returns a mono `np.float32` array normalized `/ 32768.0`, exactly like `VoiceDetector.listen_for_utterance()`. Nothing downstream may need changes.
- **Do not modify** `state_machine.py`, `memory.py`, `llm_client.py`, `transcriber.py`, or `voice_detector.py`. This feature is additive.
- **Default off:** the launch prompt defaults to No (VAD); existing behavior is unchanged unless the user opts in via the prompt or `--ptt`.
- **TDD** (write the failing test first, watch it fail, implement, watch it pass), frequent commits.
- Every commit message ends with the trailer:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```

---

### Task 1: Dependency, config, and `resolve_trigger`

Adds the dependency, the `PTT_KEY` config, and the trigger-resolution function that maps a config string to a concrete `pynput` target.

**Files:**
- Modify: `requirements.txt`
- Modify: `companion/config.py`
- Create: `companion/push_to_talk.py`
- Test: `tests/test_push_to_talk.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `resolve_trigger(key_str) -> tuple` returning `("mouse", pynput.mouse.Button)` or `("keyboard", pynput.keyboard.Key | KeyCode)`; raises `ValueError` if unbindable. Consumed by `PushToTalkRecorder` in Task 2. Also the module-level constant `_MOUSE_ALIASES`.

- [ ] **Step 1: Add the dependency**

In `requirements.txt`, add this line (keep the file's existing ordering style — append is fine):

```
pynput>=1.7,<2
```

- [ ] **Step 2: Add the config**

In `companion/config.py`, add after the VAD block (after `VAD_AGGRESSIVENESS = 2`):

```python

# Push-to-talk: hold a key/button to record instead of voice-activity detection.
# Enabled at launch (the startup prompt or --ptt); this names the trigger.
# "MOUSE_4"/"MOUSE_5" are the mouse side buttons (back/forward). Any pynput
# keyboard key name also works, e.g. "space", "ctrl_r" — useful if your mouse
# or OS does not report side buttons (see README).
PTT_KEY = "MOUSE_4"
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_push_to_talk.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `pytest tests/test_push_to_talk.py -v`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` (no `companion.push_to_talk` / `resolve_trigger` yet).

- [ ] **Step 5: Write the implementation**

Create `companion/push_to_talk.py`:

```python
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_push_to_talk.py -v`
Expected: PASS (5 passed).

- [ ] **Step 7: Commit**

```bash
git add requirements.txt companion/config.py companion/push_to_talk.py tests/test_push_to_talk.py
git commit -m "feat: add pynput dep, PTT_KEY config, and trigger resolution

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `PushToTalkRecorder`

Adds the recorder: a background `pynput` listener toggles a `threading.Event`; `listen_for_utterance()` records mic frames while the event is set and returns the same audio array shape `VoiceDetector` returns.

**Files:**
- Modify: `companion/push_to_talk.py`
- Test: `tests/test_push_to_talk.py`

**Interfaces:**
- Consumes: `resolve_trigger` from Task 1.
- Produces: `PushToTalkRecorder(sample_rate, frame_duration_ms, ptt_key)` with `listen_for_utterance() -> np.ndarray` and `close() -> None`. Consumed by `main.py` in Task 3.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_push_to_talk.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_push_to_talk.py -v`
Expected: FAIL with `ImportError` / `AttributeError` (no `PushToTalkRecorder` yet).

- [ ] **Step 3: Write the implementation**

Add to `companion/push_to_talk.py`: the new imports at the top (below the existing `from pynput import keyboard, mouse`), and the `PushToTalkRecorder` class at the end of the file.

New imports (top of file):

```python
import threading

import numpy as np
import sounddevice as sd
```

Class (end of file):

```python
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

    def listen_for_utterance(self) -> np.ndarray:
        # Block with no busy-loop until the button goes down, then record until
        # it comes up. Release latency is one frame (frame_duration_ms).
        self._pressed.wait()
        frames = []
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_size,
        ) as stream:
            while self._pressed.is_set():
                frame, _ = stream.read(self.frame_size)
                frames.append(frame)

        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames, axis=0).flatten().astype(np.float32) / 32768.0

    def close(self) -> None:
        self._listener.stop()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_push_to_talk.py -v`
Expected: PASS (9 passed — 5 from Task 1, 4 new).

- [ ] **Step 5: Commit**

```bash
git add companion/push_to_talk.py tests/test_push_to_talk.py
git commit -m "feat: add PushToTalkRecorder hold-to-record capture

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Launch selection, main wiring, and README

Adds the `ask_yes_no` helper and `--ptt` flag, selects the capture object at startup, closes the listener on exit, and documents the feature.

**Files:**
- Modify: `companion/main.py`
- Modify: `README.md`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `PushToTalkRecorder` from Task 2 (lazy import), `VoiceDetector` (existing), `config.PTT_KEY`, `config.SAMPLE_RATE`, `config.FRAME_DURATION_MS`.
- Produces: `ask_yes_no(label, default, cli_flag) -> bool`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_main.py` — extend the existing import and append the tests:

Change the import block at the top from:

```python
from companion.main import (
    PROVIDER_NAMES,
    STT_NAMES,
    choose_from_menu,
    remember_session,
)
```

to:

```python
from companion.main import (
    PROVIDER_NAMES,
    STT_NAMES,
    ask_yes_no,
    choose_from_menu,
    remember_session,
)
```

Append these tests:

```python
def test_ask_yes_no_cli_flag_skips_prompt():
    with patch("builtins.input") as mock_input:
        assert ask_yes_no("Push-to-talk?", False, True) is True
    mock_input.assert_not_called()


def test_ask_yes_no_empty_returns_default():
    with patch("builtins.input", return_value=""):
        assert ask_yes_no("Push-to-talk?", False, False) is False


def test_ask_yes_no_accepts_yes_variants():
    for value in ("y", "yes", "YES"):
        with patch("builtins.input", return_value=value):
            assert ask_yes_no("Push-to-talk?", False, False) is True


def test_ask_yes_no_treats_other_input_as_no():
    with patch("builtins.input", return_value="maybe"):
        assert ask_yes_no("Push-to-talk?", False, False) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_main.py -v`
Expected: FAIL with `ImportError: cannot import name 'ask_yes_no'`.

- [ ] **Step 3: Add the `ask_yes_no` helper**

In `companion/main.py`, add this function right after `choose_from_menu` (after line 57):

```python
def ask_yes_no(label, default, cli_flag) -> bool:
    if cli_flag:
        return True
    answer = input(f"{label} [y/N]: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")
```

- [ ] **Step 4: Run the helper tests to verify they pass**

Run: `pytest tests/test_main.py -v`
Expected: PASS (all existing tests plus the 4 new ones).

- [ ] **Step 5: Wire the flag and selection into `main()`**

In `companion/main.py`, add the `--ptt` argument. After the `--ears` argument block (after line 146), add:

```python
    parser.add_argument(
        "--ptt",
        action="store_true",
        help="enable push-to-talk (hold PTT_KEY to record) and skip the prompt",
    )
```

After the ears selection/print (after line 153, `print(f"Ears: {ears}")`), add:

```python
    ptt = ask_yes_no("Push-to-talk (hold a key to record)?", False, args.ptt)
    print(f"Push-to-talk: {'on' if ptt else 'off'}")
```

Replace the detector construction (current lines 170-176):

```python
    detector = VoiceDetector(
        sample_rate=config.SAMPLE_RATE,
        frame_duration_ms=config.FRAME_DURATION_MS,
        silence_timeout_ms=config.SILENCE_TIMEOUT_MS,
        preroll_ms=config.PREROLL_MS,
        vad_aggressiveness=config.VAD_AGGRESSIVENESS,
    )
```

with:

```python
    if ptt:
        # Lazy import: VAD-only users never load pynput. A bad PTT_KEY raises
        # here (before the session starts) with an actionable message.
        from companion.push_to_talk import PushToTalkRecorder

        capture = PushToTalkRecorder(
            config.SAMPLE_RATE, config.FRAME_DURATION_MS, config.PTT_KEY
        )
    else:
        capture = VoiceDetector(
            sample_rate=config.SAMPLE_RATE,
            frame_duration_ms=config.FRAME_DURATION_MS,
            silence_timeout_ms=config.SILENCE_TIMEOUT_MS,
            preroll_ms=config.PREROLL_MS,
            vad_aggressiveness=config.VAD_AGGRESSIVENESS,
        )
```

Update the ready hint. Replace the current line (line 188):

```python
    print(f'Ready. Say "{config.WAKE_PHRASE[0]}" to start.')
```

with:

```python
    if ptt:
        print(f'Ready. Hold your push-to-talk key and say "{config.WAKE_PHRASE[0]}" to start.')
    else:
        print(f'Ready. Say "{config.WAKE_PHRASE[0]}" to start.')
```

Change the capture call in the loop (line 192) from:

```python
            audio = detector.listen_for_utterance()
```

to:

```python
            audio = capture.listen_for_utterance()
```

Wrap the loop so the listener is stopped on exit. Change the current tail (lines 190-238):

```python
    try:
        while True:
            audio = capture.listen_for_utterance()
            ...
    except KeyboardInterrupt:
        print("\nExiting.")
```

by adding a `finally` that closes the capture if it can:

```python
    try:
        while True:
            audio = capture.listen_for_utterance()
            ...
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        # PushToTalkRecorder holds a background listener; VoiceDetector has no
        # close(). Never let cleanup mask the exit.
        close = getattr(capture, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass
```

(Leave the body of the `while` loop between lines 192-236 exactly as it is — only the call site on line 192 and the surrounding try/except/finally change.)

- [ ] **Step 6: Run the full suite**

Run: `pytest -v`
Expected: PASS (all prior tests + Task 1/2/3 additions; nothing regresses).

- [ ] **Step 7: Update the README**

In `README.md`, add a new `## Push-to-talk` section (place it after the transcription/"ears" section, before or after the Memory section — wherever run-time options are described). Content:

```markdown
## Push-to-talk

By default the companion listens continuously with voice-activity detection.
Optionally you can switch to **push-to-talk**: audio is captured only while you
hold a button, so the mic (and any cloud transcription) only ever hears what you
deliberately say.

Enable it at launch — the startup prompt asks `Push-to-talk (hold a key to
record)? [y/N]`, or pass `--ptt` to skip the prompt:

```bash
python -m companion.main --ptt
```

The wake word still applies: hold the button and say "Hey chat" to start, hold
and speak to talk, hold and say "bye bye" to end the session.

**Choosing the button.** `PTT_KEY` in `companion/config.py` names the trigger.
The default `"MOUSE_4"` is the mouse back button (`"MOUSE_5"` is forward). You can
also use any keyboard key, e.g. `"space"` or `"ctrl_r"`.

**macOS:** the first time push-to-talk runs, macOS asks for **Accessibility**
permission (System Settings → Privacy & Security → Accessibility) so the app can
read the button globally. Until you grant it, macOS silently drops the events —
if you hold the button and nothing records, that permission is the likely cause.
If your mouse or OS does not report the side buttons at all, set `PTT_KEY` to a
keyboard key instead.
```

- [ ] **Step 8: Commit**

```bash
git add companion/main.py tests/test_main.py README.md
git commit -m "feat: select push-to-talk at launch and wire capture

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §1 dependency → Task 1 Step 1. §3 config → Task 1 Step 2. §5 resolution → Task 1 Steps 3-6.
- §4 recorder → Task 2. §6 main wiring (`ask_yes_no`, `--ptt`, selection, ready hint, `close()` on exit) → Task 3 Steps 1-6. §7 macOS/README → Task 3 Step 7.
- §2 selection (prompt default No, `--ptt` skips) → Task 3 Steps 1, 3, 5. §8 testing → the test steps in every task.
- Out-of-scope items (no VAD removal, no toggle, no save-on-exit change) are respected: `VoiceDetector` untouched, hold-only semantics, memory path unchanged.

**Placeholder scan:** none — every step has concrete code/commands and expected output.

**Type consistency:** `resolve_trigger` returns `(kind, target)` in Task 1 and is destructured the same way in Task 2's `__init__`. `listen_for_utterance()`/`close()` signatures match between Task 2 and Task 3's usage. `ask_yes_no(label, default, cli_flag)` matches between Task 3 Steps 1 and 3. The capture variable is renamed `detector` → `capture` consistently at construction, call site, and cleanup.
