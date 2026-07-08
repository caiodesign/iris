# Push-to-Talk — Design Spec

**Date:** 2026-07-08
**Status:** Approved
**Feature:** Optional push-to-talk capture. Independent of Features A (swappable
STT), B (structured memory), and C (conversation coach, not yet built).

## Problem

Capture today is always voice-activity detection (VAD): `VoiceDetector.listen_for_utterance()`
opens the mic, waits for speech, and returns the utterance after a silence
timeout. The mic is effectively always live while the script runs.

Two consequences the user hit in practice:

1. **The ears never sleep.** With cloud ears (OpenAI), *every* utterance the VAD
   picks up is uploaded to OpenAI for transcription — even while the state machine
   is `ASLEEP` — purely so the app can string-match it against the wake phrase and
   almost always discard it. That is a continuous privacy and cost drain from
   background speech.
2. **No deliberate control over when the mic is hot.** VAD triggers on any nearby
   speech; the user cannot choose the exact moment audio is captured.

**North star:** deliberate capture. The user holds a button to talk, so the only
audio ever captured — and the only audio ever sent to the ears — is what the user
intentionally speaks.

## Core idea: push-to-talk is a pure capture swap

Push-to-talk (PTT) replaces *how audio is captured*, and nothing else. When PTT is
enabled, holding a configured button (default the mouse back button, `MOUSE_4`)
records for exactly as long as it is held; releasing returns the utterance. The
returned audio is the same `np.ndarray` (mono, 16 kHz, float32, `/32768.0`) that
VAD returns today, so everything downstream is byte-for-byte identical:

- The **wake phrase stays.** You hold the button and say "Hey chat" to wake, hold
  and speak to talk, hold and say "bye bye" to sleep. The wake word is a second,
  deliberate gate on top of the physical button — and keeping it means the
  `StateMachine`, memory, and the main loop body do not change at all.
- The **state machine, wake/sleep/cancel logic, and memory are untouched.** PTT is
  not a new mode with its own branches; it is a different object behind the same
  `listen_for_utterance()` call.

```
Hold MOUSE_4, say "Hey chat"     → wakes (loads memory, greets)
Hold MOUSE_4, say your message   → forwarded to the brain
Hold MOUSE_4, say "cancel that"  → discards it
Hold MOUSE_4, say "bye bye"      → saves memory, back to sleep
```

The one accepted cost: starting a session takes one extra deliberate step (hold +
say "Hey chat", then hold again to talk). In exchange, PTT adds zero new control
flow and captures only intentional audio.

## Decisions

### 1. New dependency: `pynput`

Reading a mouse side-button *while the terminal is not focused* (the whole point —
the user is doing other things on their PC) requires a global input hook. Terminal
stdin cannot see mouse button 4. `pynput` is the standard cross-platform library
for global mouse/keyboard listeners.

- Add `pynput>=1.7,<2` to `requirements.txt`.
- This is the first library in the project that hooks system input. On **macOS**
  it triggers a one-time **Accessibility permission** prompt the first time the
  listener runs; until granted, the OS silently withholds events. This is
  documented in the README (see §7), not detected programmatically — macOS gives
  no reliable signal that permission is missing.
- `pynput` is imported lazily, only when PTT is enabled, so VAD-only users never
  load it and any import/platform issue surfaces only for users who opt in.

### 2. Selection at launch

PTT is chosen at startup, alongside brain and ears, before models load.

- **Menu prompt** (only when `--ptt` is absent): a yes/no question defaulting to
  **No** (VAD), so existing behavior is unchanged unless the user opts in.
  ```
  Push-to-talk (hold a key to record)? [y/N]:
  ```
- **CLI flag:** `--ptt` (argparse `store_true`) enables PTT and skips the prompt,
  mirroring how `--brain`/`--ears` skip their menus. There is no `--no-ptt`;
  pressing Enter at the prompt selects VAD.
- A new helper `ask_yes_no(label, default, cli_flag)` handles the prompt; it lives
  in `main.py` next to `choose_from_menu` and is unit-tested the same way.

### 3. Config additions (`config.py`)

```python
# Push-to-talk: hold a key/button to record instead of voice-activity detection.
# Enabled at launch (the startup prompt or --ptt); this names the trigger.
# "MOUSE_4"/"MOUSE_5" are the mouse side buttons (back/forward). Any pynput
# keyboard key name also works, e.g. "space", "ctrl_r" — useful if your mouse
# or OS does not report side buttons (see README).
PTT_KEY = "MOUSE_4"
```

No other config changes. VAD settings (`SILENCE_TIMEOUT_MS`, `PREROLL_MS`,
`VAD_AGGRESSIVENESS`) remain and are simply unused while PTT is active.

### 4. Capture module (`companion/push_to_talk.py`)

A new `PushToTalkRecorder` exposing the **same interface** as `VoiceDetector`:

```python
PushToTalkRecorder(sample_rate, frame_duration_ms, ptt_key)
    listen_for_utterance() -> np.ndarray   # blocks until the button is pressed,
                                           # records while held, returns on release
    close() -> None                        # stops the background listener
```

Design:

- A persistent `pynput` listener runs on a background daemon thread and toggles a
  `threading.Event` (`_pressed`) via `on_press`/`on_release` callbacks (which are
  methods, so tests can call them directly without a real listener).
- `listen_for_utterance()`:
  1. `self._pressed.wait()` — block with no busy-loop until the button goes down.
  2. Open an `sd.InputStream` and read `frame_size` chunks in a loop **while**
     `self._pressed.is_set()`; append each chunk.
  3. On release, close the stream and return
     `np.concatenate(frames).flatten().astype(np.float32) / 32768.0`.
- No preroll ring buffer is needed — the user controls the start, so there is no
  first-syllable clipping to compensate for. Simpler than the VAD path.
- Release latency is one frame (`FRAME_DURATION_MS`, 30 ms) — imperceptible.
- If the button is tapped and released before any frame is read, the returned
  array is empty; `transcribe` yields `""` and the main loop's existing
  `if not text: continue` guard drops it. No special case.

### 5. Trigger resolution (`resolve_trigger`)

A module-level `resolve_trigger(key_str) -> tuple` maps the config string to a
concrete `pynput` target and tells the recorder which listener to start:

- **Mouse buttons.** `MOUSE_LEFT`/`MOUSE_RIGHT`/`MOUSE_MIDDLE` map to
  `mouse.Button.left`/`right`/`middle`. `MOUSE_4`/`MOUSE_5` (back/forward side
  buttons) are platform-specific `Button` members — resolved by trying known
  attribute names in order (`MOUSE_4` → `button8`, then `x1`; `MOUSE_5` →
  `button9`, then `x2`) and using the first that exists.
- **Keyboard keys.** Any other string is resolved against the keyboard: a named
  key via `keyboard.Key[name]` (e.g. `space`), falling back to
  `keyboard.KeyCode.from_char(name)` for single characters.
- **Failure.** If nothing resolves (e.g. a Mac that does not expose the side
  button), raise `ValueError` with an actionable message:
  `Could not bind PTT key 'MOUSE_4' on this platform. Set PTT_KEY in
  companion/config.py to a keyboard key such as "space".`

Returns `("mouse", Button)` or `("keyboard", key)` so the recorder starts the
matching `pynput` listener and its `on_press`/`on_release` compare against the
resolved target.

### 6. Main wiring (`main.py`)

- Parse `--ptt`; ask `ask_yes_no` if the flag is absent. Print the choice
  (`Push-to-talk: on/off`) like brain/ears.
- Build the capture object once, before the loop:
  ```python
  if ptt:
      from companion.push_to_talk import PushToTalkRecorder
      capture = PushToTalkRecorder(config.SAMPLE_RATE, config.FRAME_DURATION_MS, config.PTT_KEY)
  else:
      capture = VoiceDetector(...)   # exactly as today
  ```
  Resolution/binding happens here, so a bad `PTT_KEY` fails fast at launch with the
  §5 message, before models load.
- The loop body is unchanged except the call site reads
  `audio = capture.listen_for_utterance()`. (`detector` is renamed to `capture`;
  no other change.)
- The ready hint reflects the mode: PTT prints
  `Ready. Hold your push-to-talk key and say "Hey chat" to start.`; VAD keeps its
  current message.
- On exit (`KeyboardInterrupt`), call `capture.close()` if it exists so the daemon
  listener stops cleanly. Wrapped so it never masks the exit.

### 7. Resilience & safety

- **Fail fast on an unbindable key** — resolution runs at construction, before the
  session starts (§5).
- **Daemon listener** — the `pynput` listener thread is a daemon, so it never
  blocks process exit even if `close()` is missed.
- **macOS Accessibility** — README documents that the first PTT run prompts for
  Accessibility permission and that events are silently dropped until it is
  granted. A short troubleshooting note ("held the button and nothing recorded →
  grant Accessibility permission") is included.
- **No downstream risk** — because PTT returns the identical audio array, the
  existing transcription, state-machine, LLM, and memory guards already cover
  everything after capture. PTT adds no new failure surface past the mic.

### 8. Testing

pytest + `unittest.mock`, mocking `pynput` and `sounddevice` — same style as
`test_transcriber.py` / `test_main.py`, no real devices or hooks:

- **`resolve_trigger`**: `"MOUSE_4"` → `("mouse", <button>)`; `"space"` →
  `("keyboard", <key>)`; an unresolvable name raises `ValueError`.
- **`PushToTalkRecorder.listen_for_utterance`**: with `sd.InputStream` mocked to
  yield fixed frames, driving `_pressed` (set, then cleared after N reads via the
  mock's `side_effect`) returns a float32 array that concatenates exactly the
  frames captured while pressed, normalized by 32768.
- **Press/release callbacks**: `on_press`/`on_release` for the matching target set
  and clear `_pressed`; a non-matching button/key is ignored.
- **`close()`** stops the listener (mock listener's `stop` called).
- **`ask_yes_no`** (in `test_main.py`): `--ptt` flag returns `True` without
  prompting; prompt `"y"`/`"yes"` → `True`; empty input → default `False`;
  garbage → default.

## Out of scope

- Replacing or removing VAD — it stays as the default capture path.
- A visual/GUI indicator; terminal prints only.
- Toggle (press-once-to-start, press-again-to-stop) behavior — PTT is strictly
  hold-to-talk, as specified.
- Programmatic detection of macOS Accessibility permission (not reliably possible).
- Save-on-exit for memory — unchanged; "bye bye" still saves exactly as today.
- Rebinding the key at runtime; the key is chosen via `config.py` / launch only.
