# Web Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A dark, Claude.ai-style web page (served by the app itself) that starts/ends voice sessions, shows the conversation as live chat, and displays the memory files — replacing the terminal prompts and log lines.

**Architecture:** Extract the listen→transcribe→think→speak loop out of `main.py` into `companion/session.py`, which reports everything through an `emit(event_dict)` callback and honors a `should_stop()` flag. `companion/server.py` (FastAPI) runs one session in a background thread, fans events out to browsers over a WebSocket, and serves a static page from `companion/web/`. `main.py` keeps working as the terminal front-end using a print-based emitter.

**Tech Stack:** Python 3.13, FastAPI + uvicorn, pytest, plain HTML/CSS/JS (no Node.js, no build step, no CDN).

**Spec:** `docs/superpowers/specs/2026-07-08-web-interface-design.md`

## Global Constraints

- No new JS toolchain: `companion/web/` is plain static files, self-contained, no CDN references.
- Server binds `127.0.0.1` port `8000` only; URL is `http://localhost:8000`.
- New Python deps pinned in the existing style: `fastapi>=0.110,<1`, `uvicorn>=0.30,<1`.
- Terminal mode (`python -m companion.main`) must keep working with its current flags (`--brain`, `--ears`, `--ptt`) and menus.
- Event dicts are the exact WebSocket protocol from the spec: `{"event": "status", "state": ...}`, `{"event": "heard"|"reply"|"system"|"warning"|"error", "text": ...}`, `{"event": "session_ended"}`.
- All existing tests must keep passing. Run with: `python -m pytest tests/ -q` from the repo root.
- Windows dev box; use forward-slash paths in code (`os.path.join` handles it).

---

### Task 1: `stop_check` support in both capture classes

The End-session button must be able to interrupt a blocking `listen_for_utterance()`. Both capture classes gain an optional `stop_check` callable, checked every frame/wait tick; when it returns True, `listen_for_utterance` returns `None` and the caller discards the partial audio.

**Files:**
- Modify: `companion/voice_detector.py`
- Modify: `companion/push_to_talk.py`
- Create: `tests/test_voice_detector.py`
- Modify: `tests/test_push_to_talk.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `VoiceDetector.listen_for_utterance(stop_check=None) -> np.ndarray | None` and `PushToTalkRecorder.listen_for_utterance(stop_check=None) -> np.ndarray | None`. `stop_check` is a zero-arg callable returning bool. Return `None` means "stop was requested, no utterance". Calling with no argument behaves exactly as today.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_voice_detector.py`:

```python
# tests/test_voice_detector.py
import types
from unittest.mock import patch

import numpy as np

from companion.voice_detector import VoiceDetector


def _silent_detector():
    detector = VoiceDetector(
        sample_rate=16000,
        frame_duration_ms=30,
        silence_timeout_ms=2000,
        preroll_ms=300,
        vad_aggressiveness=2,
    )
    # webrtcvad.Vad is a C extension; swap the whole attribute instead of
    # patching a method on it.
    detector.vad = types.SimpleNamespace(is_speech=lambda data, rate: False)
    return detector


def test_listen_returns_none_when_stop_requested():
    detector = _silent_detector()
    frame = np.zeros((480, 1), dtype=np.int16)
    calls = {"n": 0}

    def stop_check():
        calls["n"] += 1
        return calls["n"] >= 3

    with patch("companion.voice_detector.sd.InputStream") as MockStream:
        MockStream.return_value.__enter__.return_value.read.return_value = (
            frame,
            False,
        )
        result = detector.listen_for_utterance(stop_check=stop_check)

    assert result is None


def test_listen_without_stop_check_still_captures_speech():
    detector = _silent_detector()
    frame = np.ones((480, 1), dtype=np.int16) * 100
    # Speech for 2 frames, then silence until the timeout trips.
    speech = {"n": 0}

    def is_speech(data, rate):
        speech["n"] += 1
        return speech["n"] <= 2

    detector.vad = types.SimpleNamespace(is_speech=is_speech)
    with patch("companion.voice_detector.sd.InputStream") as MockStream:
        MockStream.return_value.__enter__.return_value.read.return_value = (
            frame,
            False,
        )
        audio = detector.listen_for_utterance()

    assert audio is not None
    assert audio.dtype == np.float32
    assert len(audio) > 0
```

Append to `tests/test_push_to_talk.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_voice_detector.py tests/test_push_to_talk.py -q`
Expected: the new tests FAIL with `TypeError: listen_for_utterance() got an unexpected keyword argument 'stop_check'`; all pre-existing tests still pass.

- [ ] **Step 3: Implement `stop_check` in `VoiceDetector`**

In `companion/voice_detector.py`, replace the `listen_for_utterance` method:

```python
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
```

- [ ] **Step 4: Implement `stop_check` in `PushToTalkRecorder`**

In `companion/push_to_talk.py`, replace the `listen_for_utterance` method:

```python
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
```

- [ ] **Step 5: Run the full suite to verify it passes**

Run: `python -m pytest tests/ -q`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add companion/voice_detector.py companion/push_to_talk.py tests/test_voice_detector.py tests/test_push_to_talk.py
git commit -m "feat: capture classes accept a stop_check to interrupt listening"
```

---

### Task 2: Session engine — event-emitting conversation loop

Create `companion/session.py` with the conversation loop extracted from `main.py`, decoupled from printing: every observable moment becomes an event dict passed to `emit`.

**Files:**
- Create: `companion/session.py`
- Create: `tests/test_session.py`

**Interfaces:**
- Consumes: `StateMachine.process(text) -> Action`, `State.ACTIVE` (from `companion/state_machine.py`); `capture.listen_for_utterance(stop_check=None)` from Task 1; `LLMClient` methods `reset(memory)`, `seed_assistant(text)`, `send(text)`, `summarize(instruction)`, `has_user_turns()`; `Memory` methods `load()`, `load_durable()`, `append_timeline(entry)`, `write_durable(text)`; `config.GREETING`, `config.TIMELINE_PROMPT`, `config.DURABLE_MERGE_PROMPT`.
- Produces (used by Tasks 3–5):
  - `PROVIDER_NAMES: list[str]` = `["local", "claude", "openai", "zai"]`
  - `STT_NAMES: list[str]` = `["local", "openai"]`
  - `run_loop(capture, transcriber, llm, memory, speaker, machine, emit, should_stop) -> None` — `emit` takes one dict; `should_stop` is a zero-arg callable returning bool.
  - `remember_session(llm, memory, emit) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_session.py`:

```python
# tests/test_session.py
from unittest.mock import MagicMock

import numpy as np

from companion import config, session
from companion.state_machine import StateMachine


class FakeCapture:
    """Yields `count` dummy audio arrays, then flags itself exhausted so the
    should_stop wired in run_scripted ends the loop."""

    def __init__(self, count):
        self.remaining = count
        self.exhausted = False

    def listen_for_utterance(self, stop_check=None):
        if self.remaining == 0:
            self.exhausted = True
            return np.zeros(0, dtype=np.float32)
        self.remaining -= 1
        return np.ones(160, dtype=np.float32)


class FakeTranscriber:
    """Returns scripted texts in order; an Exception item is raised instead."""

    def __init__(self, texts):
        self.texts = list(texts)

    def transcribe(self, audio):
        item = self.texts.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeLLM:
    def __init__(self, reply="Nice!"):
        self.reply = reply
        self.reset_memory = None
        self.seeded = []
        self.sent = []
        self.summaries = ["- timeline entry", "## Facts\n- a fact"]
        self._has_user = False

    def reset(self, memory=""):
        self.reset_memory = memory

    def seed_assistant(self, text):
        self.seeded.append(text)

    def send(self, text):
        self.sent.append(text)
        if isinstance(self.reply, Exception):
            raise self.reply
        self._has_user = True
        return self.reply

    def summarize(self, instruction):
        item = self.summaries.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def has_user_turns(self):
        return self._has_user


class FakeSpeaker:
    def __init__(self, fail=False):
        self.fail = fail
        self.spoken = []

    def speak(self, text):
        if self.fail:
            raise RuntimeError("no audio device")
        self.spoken.append(text)


def run_scripted(texts, llm=None, speaker=None, memory=None):
    """Drive run_loop through the scripted utterances, then stop."""
    events = []
    capture = FakeCapture(len(texts))
    transcriber = FakeTranscriber(texts)
    llm = llm if llm is not None else FakeLLM()
    speaker = speaker if speaker is not None else FakeSpeaker()
    if memory is None:
        memory = MagicMock()
        memory.load.return_value = "remembered stuff"
        memory.load_durable.return_value = ""
    session.run_loop(
        capture,
        transcriber,
        llm,
        memory,
        speaker,
        StateMachine(),
        events.append,
        lambda: capture.exhausted,
    )
    return events, llm, speaker, memory


def texts_of(events, kind):
    return [e["text"] for e in events if e["event"] == kind]


def states_of(events):
    return [e["state"] for e in events if e["event"] == "status"]


def test_wake_greets_with_seeded_history_and_memory():
    events, llm, speaker, memory = run_scripted(["hey chat"])
    assert "Waking up." in texts_of(events, "system")
    assert texts_of(events, "reply") == [config.GREETING]
    # The script ending counts as a stop while awake, so "Bye for now!"
    # follows the greeting (that path has its own test below).
    assert speaker.spoken[0] == config.GREETING
    assert llm.reset_memory == "remembered stuff"
    assert llm.seeded == [config.GREETING]


def test_ignored_speech_while_asleep_emits_nothing():
    events, llm, speaker, _ = run_scripted(["just background noise"])
    assert texts_of(events, "heard") == []
    assert texts_of(events, "reply") == []
    assert speaker.spoken == []
    assert llm.sent == []


def test_forward_emits_heard_and_reply_and_speaks():
    events, llm, speaker, _ = run_scripted(["hey chat", "I like ramen"])
    assert texts_of(events, "heard") == ["I like ramen"]
    assert "Nice!" in texts_of(events, "reply")
    assert "Nice!" in speaker.spoken
    assert llm.sent == ["I like ramen"]


def test_status_cycles_listening_thinking_speaking():
    events, _, _, _ = run_scripted(["hey chat", "I like ramen"])
    states = states_of(events)
    assert states[0] == "listening"
    assert "thinking" in states
    assert "speaking" in states


def test_cancel_discards_without_sending():
    events, llm, _, _ = run_scripted(["hey chat", "cancel that"])
    assert "Discarded that." in texts_of(events, "system")
    assert llm.sent == []


def test_sleep_says_goodbye_and_remembers():
    events, llm, speaker, memory = run_scripted(
        ["hey chat", "I like ramen", "bye bye"]
    )
    assert "Going back to sleep." in texts_of(events, "system")
    assert "Bye for now!" in speaker.spoken
    memory.append_timeline.assert_called_once_with("- timeline entry")
    memory.write_durable.assert_called_once_with("## Facts\n- a fact")


def test_stop_while_awake_runs_the_goodbye_path():
    # The script ends (End button) while the machine is still ACTIVE.
    events, llm, speaker, memory = run_scripted(["hey chat", "I like ramen"])
    assert speaker.spoken[-1] == "Bye for now!"
    memory.append_timeline.assert_called_once()
    memory.write_durable.assert_called_once()


def test_stop_while_asleep_skips_the_goodbye():
    events, llm, speaker, memory = run_scripted(["hey chat", "bye bye"])
    # One goodbye from the stop phrase, no second one at loop exit.
    assert speaker.spoken.count("Bye for now!") == 1


def test_transcription_failure_warns_and_continues():
    events, llm, speaker, _ = run_scripted(
        [RuntimeError("boom"), "hey chat"]
    )
    warnings = texts_of(events, "warning")
    assert any("Transcription failed" in w for w in warnings)
    assert texts_of(events, "reply") == [config.GREETING]


def test_llm_failure_warns_and_speaks_recovery_line():
    llm = FakeLLM(reply=RuntimeError("api down"))
    events, llm, speaker, _ = run_scripted(["hey chat", "hello there"], llm=llm)
    warnings = texts_of(events, "warning")
    assert any("brain failed to reply" in w for w in warnings)
    assert "Sorry, I had trouble thinking. Say that again?" in speaker.spoken


def test_tts_failure_warns_and_continues():
    speaker = FakeSpeaker(fail=True)
    events, _, _, _ = run_scripted(["hey chat"], speaker=speaker)
    warnings = texts_of(events, "warning")
    assert any("Text-to-speech playback failed" in w for w in warnings)


def test_remember_session_skips_without_user_turns():
    llm = MagicMock()
    llm.has_user_turns.return_value = False
    memory = MagicMock()
    events = []
    session.remember_session(llm, memory, events.append)
    llm.summarize.assert_not_called()
    memory.append_timeline.assert_not_called()
    memory.write_durable.assert_not_called()


def test_remember_session_still_merges_durable_when_timeline_fails():
    llm = MagicMock()
    llm.has_user_turns.return_value = True
    llm.summarize.side_effect = [Exception("blip"), "## Facts\n- x"]
    memory = MagicMock()
    memory.load_durable.return_value = ""
    events = []
    session.remember_session(llm, memory, events.append)
    memory.append_timeline.assert_not_called()
    memory.write_durable.assert_called_once_with("## Facts\n- x")
    assert any(e["event"] == "warning" for e in events)


def test_remember_session_keeps_timeline_when_durable_fails():
    llm = MagicMock()
    llm.has_user_turns.return_value = True
    llm.summarize.side_effect = ["- entry", Exception("blip")]
    memory = MagicMock()
    memory.load_durable.return_value = ""
    events = []
    session.remember_session(llm, memory, events.append)
    memory.append_timeline.assert_called_once_with("- entry")
    memory.write_durable.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_session.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'companion.session'` (collection error).

- [ ] **Step 3: Create `companion/session.py`**

```python
# companion/session.py
"""Session engine: the listen->transcribe->think->speak loop, decoupled from
any display. Callers pass emit(event_dict) and should_stop() and render the
events however they like — the terminal prints them, the web page streams
them over a WebSocket. Event shapes are the WebSocket protocol documented in
docs/superpowers/specs/2026-07-08-web-interface-design.md."""

from companion import config
from companion.state_machine import Action, State

PROVIDER_NAMES = ["local", "claude", "openai", "zai"]
STT_NAMES = ["local", "openai"]


def _speak(speaker, text, emit) -> None:
    emit({"event": "status", "state": "speaking"})
    try:
        speaker.speak(text)
    except Exception as exc:
        emit(
            {
                "event": "warning",
                "text": f"Text-to-speech playback failed ({exc}). "
                "Continuing without audio.",
            }
        )


def remember_session(llm, memory, emit) -> None:
    if not llm.has_user_turns():
        return
    emit({"event": "system", "text": "Remembering this session..."})
    # Two independent side-channel calls: a failure in one must not skip the
    # other, and neither may crash the goodbye (mirrors the send/tts guards).
    try:
        memory.append_timeline(llm.summarize(config.TIMELINE_PROMPT))
    except Exception as exc:
        emit({"event": "warning", "text": f"Could not update timeline memory ({exc})."})
    try:
        merged = llm.summarize(config.DURABLE_MERGE_PROMPT + memory.load_durable())
        memory.write_durable(merged)
    except Exception as exc:
        emit({"event": "warning", "text": f"Could not update durable memory ({exc})."})


def run_loop(capture, transcriber, llm, memory, speaker, machine, emit, should_stop) -> None:
    while not should_stop():
        emit({"event": "status", "state": "listening"})
        audio = capture.listen_for_utterance(stop_check=should_stop)
        if should_stop():
            break
        if audio is None or len(audio) == 0:
            continue
        emit({"event": "status", "state": "thinking"})
        try:
            text = transcriber.transcribe(audio)
        except Exception as exc:
            # Cloud STT can fail on a network blip; a local frame can be bad.
            # Drop this utterance and keep listening instead of crashing.
            emit({"event": "warning", "text": f"Transcription failed ({exc})."})
            continue
        if not text:
            continue

        action = machine.process(text)
        if action == Action.IGNORE:
            continue
        elif action == Action.WAKE:
            emit({"event": "system", "text": "Waking up."})
            # Fresh history per session, and the greeting is seeded as an
            # assistant turn — otherwise the LLM doesn't know the opening
            # question was already asked and re-asks it.
            llm.reset(memory.load())
            llm.seed_assistant(config.GREETING)
            emit({"event": "reply", "text": config.GREETING})
            _speak(speaker, config.GREETING, emit)
        elif action == Action.CANCEL:
            emit({"event": "system", "text": "Discarded that."})
        elif action == Action.SLEEP:
            emit({"event": "system", "text": "Going back to sleep."})
            _speak(speaker, "Bye for now!", emit)
            remember_session(llm, memory, emit)
        elif action == Action.FORWARD:
            emit({"event": "heard", "text": text})
            try:
                reply = llm.send(text)
            except Exception as exc:
                # Cloud APIs hiccup and Ollama can die mid-session; keep the
                # session alive. (The dangling user turn is harmless: every
                # provider accepts consecutive user messages.)
                emit({"event": "warning", "text": f"The brain failed to reply ({exc})."})
                _speak(speaker, "Sorry, I had trouble thinking. Say that again?", emit)
                continue
            emit({"event": "reply", "text": reply})
            _speak(speaker, reply, emit)

    # Stopped by flag (End button / Ctrl+C path) while a conversation was
    # active: run the same goodbye as the stop phrase so memory is never
    # silently dropped.
    if machine.state == State.ACTIVE:
        _speak(speaker, "Bye for now!", emit)
        remember_session(llm, memory, emit)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_session.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add companion/session.py tests/test_session.py
git commit -m "feat: extract event-emitting session loop into companion/session.py"
```

---

### Task 3: Session engine — preflight checks and `run_session`

Add the startup half: preflight checks that return error strings (no `sys.exit`), component builders, and `run_session` which ties preflight + load + loop + cleanup together.

**Files:**
- Modify: `companion/session.py`
- Modify: `tests/test_session.py` (append tests)

**Interfaces:**
- Consumes: `run_loop` and `remember_session` from Task 2; `make_provider(brain)`, `REQUIRED_ENV` (dict provider->env var) from `companion/providers.py`; `make_transcriber(name)` from `companion/transcriber.py`; `VoiceDetector`, `PushToTalkRecorder`, `LLMClient`, `Memory`, `Speaker`, `StateMachine`.
- Produces (used by Tasks 4–5):
  - `preflight_error(brain: str, ears: str) -> str | None` — actionable message, or None if all checks pass.
  - `build_capture(ptt: bool)` — returns a capture object; may raise (bad PTT key).
  - `load_transcriber(ears: str)` — returns a transcriber; raises `RuntimeError` with an actionable message on GPU/model failure.
  - `run_session(brain, ears, ptt, emit, should_stop) -> bool` — False if the session could not start (an `error` event was emitted), True on a clean end. Never raises except `KeyboardInterrupt`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_session.py`:

```python
import os

import pytest


def test_preflight_reports_unreachable_ollama(monkeypatch):
    def boom():
        raise ConnectionError("connection refused")

    monkeypatch.setattr(session.ollama, "list", boom)
    error = session.preflight_error("local", "local")
    assert error is not None
    assert "Ollama" in error


def test_preflight_reports_missing_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    error = session.preflight_error("claude", "local")
    assert error is not None
    assert "ANTHROPIC_API_KEY" in error


def test_preflight_reports_missing_openai_key_for_cloud_ears(monkeypatch):
    monkeypatch.setattr(session.ollama, "list", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    error = session.preflight_error("local", "openai")
    assert error is not None
    assert "OPENAI_API_KEY" in error


def test_preflight_reports_missing_tts_files(monkeypatch):
    monkeypatch.setattr(session.ollama, "list", lambda: None)

    class OkStream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(session.sd, "InputStream", lambda **kwargs: OkStream())
    monkeypatch.setattr(config, "KOKORO_MODEL_PATH", "does-not-exist.onnx")
    error = session.preflight_error("local", "local")
    assert error is not None
    assert "Kokoro" in error


def test_run_session_emits_error_and_returns_false_on_preflight_failure(
    monkeypatch,
):
    events = []
    monkeypatch.setattr(session, "preflight_error", lambda b, e: "Ollama is down")
    ok = session.run_session("local", "local", False, events.append, lambda: True)
    assert ok is False
    assert events == [{"event": "error", "text": "Ollama is down"}]


def test_run_session_emits_error_when_a_component_fails_to_load(monkeypatch):
    events = []
    monkeypatch.setattr(session, "preflight_error", lambda b, e: None)

    def bad_capture(ptt):
        raise RuntimeError("Could not bind PTT key")

    monkeypatch.setattr(session, "build_capture", bad_capture)
    ok = session.run_session("claude", "local", True, events.append, lambda: True)
    assert ok is False
    assert events[-1] == {"event": "error", "text": "Could not bind PTT key"}
    # It got as far as loading before failing.
    assert {"event": "status", "state": "loading"} in events


def test_run_session_happy_path_runs_loop_and_closes_capture(monkeypatch):
    events = []
    closed = {"done": False}

    class DummyCapture:
        def close(self):
            closed["done"] = True

    monkeypatch.setattr(session, "preflight_error", lambda b, e: None)
    monkeypatch.setattr(session, "build_capture", lambda ptt: DummyCapture())
    monkeypatch.setattr(session, "load_transcriber", lambda ears: object())
    monkeypatch.setattr(session, "make_provider", lambda brain: object())
    monkeypatch.setattr(
        session, "Speaker", lambda *args: FakeSpeaker()
    )
    loop_ran = {"done": False}

    def fake_loop(capture, transcriber, llm, memory, speaker, machine, emit, should_stop):
        loop_ran["done"] = True

    monkeypatch.setattr(session, "run_loop", fake_loop)
    ok = session.run_session("local", "local", False, events.append, lambda: True)
    assert ok is True
    assert loop_ran["done"] is True
    assert closed["done"] is True
    ready_lines = [e["text"] for e in events if e["event"] == "system"]
    assert any(t.startswith("Ready.") for t in ready_lines)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_session.py -q`
Expected: the new tests FAIL with `AttributeError: module 'companion.session' has no attribute 'preflight_error'` (and similar); Task 2 tests still pass.

- [ ] **Step 3: Implement preflight and `run_session`**

In `companion/session.py`, extend the imports at the top to:

```python
import os

import numpy as np
import ollama
import sounddevice as sd

from companion import config
from companion.llm_client import LLMClient
from companion.memory import Memory
from companion.providers import REQUIRED_ENV, make_provider
from companion.speaker import Speaker
from companion.state_machine import Action, State, StateMachine
from companion.transcriber import make_transcriber
from companion.voice_detector import VoiceDetector
```

Then add below `PROVIDER_NAMES` / `STT_NAMES`:

```python
def _missing_key_error(name: str) -> "str | None":
    env_var = REQUIRED_ENV[name]
    if not os.environ.get(env_var):
        return (
            f"{env_var} is not set. Put it in a .env file at the project root "
            '(see the README section "Cloud brains") and try again.'
        )
    return None


def preflight_error(brain: str, ears: str) -> "str | None":
    """Return an actionable error message if a session cannot start, else None.

    Same checks main.py used to sys.exit() on, reworded for a UI banner."""
    if brain == "local":
        # Cloud brains never touch Ollama, so the llama model stays out of
        # VRAM — that's the point of using them while gaming.
        try:
            ollama.list()
        except Exception as exc:
            return f"Could not reach Ollama ({exc}). Is it running?"
    else:
        error = _missing_key_error(brain)
        if error:
            return error
    if ears == "openai":
        # OpenAI cloud ears reuse the brain's OPENAI_API_KEY; fail fast if
        # absent (REQUIRED_ENV["openai"] == "OPENAI_API_KEY").
        error = _missing_key_error("openai")
        if error:
            return error
    try:
        with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1, dtype="int16"):
            pass
    except Exception as exc:
        return f"Could not access a microphone ({exc})."
    missing = [
        path
        for path in (config.KOKORO_MODEL_PATH, config.KOKORO_VOICES_PATH)
        if not os.path.exists(path)
    ]
    if missing:
        return (
            f"Kokoro TTS file(s) not found: {', '.join(missing)}. "
            "Run README step 4 from the project root to download them."
        )
    return None


def build_capture(ptt: bool):
    if ptt:
        # Lazy import: VAD-only users never load pynput. A bad PTT_KEY raises
        # here (before the loop starts) with an actionable message.
        from companion.push_to_talk import PushToTalkRecorder

        return PushToTalkRecorder(
            config.SAMPLE_RATE, config.FRAME_DURATION_MS, config.PTT_KEY
        )
    return VoiceDetector(
        sample_rate=config.SAMPLE_RATE,
        frame_duration_ms=config.FRAME_DURATION_MS,
        silence_timeout_ms=config.SILENCE_TIMEOUT_MS,
        preroll_ms=config.PREROLL_MS,
        vad_aggressiveness=config.VAD_AGGRESSIVENESS,
    )


def load_transcriber(ears: str):
    if ears == "openai":
        # Cloud ears: no GPU model to load, so no CUDA warm-up. The key was
        # already checked in preflight.
        return make_transcriber("openai")
    try:
        transcriber = make_transcriber("local")
        # CUDA libraries load lazily on the first transcription, not at model
        # construction — warm up now so GPU problems surface here, with the
        # hint below, instead of as a traceback mid-conversation.
        transcriber.transcribe(np.zeros(config.SAMPLE_RATE, dtype=np.float32))
        return transcriber
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the Whisper model on '{config.WHISPER_DEVICE}' "
            f"({exc}). Hint: GPU mode needs NVIDIA libraries (see README step "
            '3). To run on CPU instead, set WHISPER_DEVICE = "cpu" and '
            'WHISPER_COMPUTE_TYPE = "int8" in companion/config.py.'
        ) from exc


def _close_capture(capture) -> None:
    # PushToTalkRecorder holds a background listener; VoiceDetector has no
    # close(). Never let cleanup mask the exit.
    close = getattr(capture, "close", None)
    if close is not None:
        try:
            close()
        except Exception:
            pass


def run_session(brain, ears, ptt, emit, should_stop) -> bool:
    """Preflight, load, loop, cleanup. Returns False if the session could not
    start (an "error" event was emitted), True on a clean end."""
    error = preflight_error(brain, ears)
    if error:
        emit({"event": "error", "text": error})
        return False

    emit({"event": "status", "state": "loading"})
    emit({"event": "system", "text": "Loading models (this may take a moment)..."})
    capture = None
    try:
        capture = build_capture(ptt)
        transcriber = load_transcriber(ears)
        llm = LLMClient(make_provider(brain), config.SYSTEM_PROMPT)
        memory = Memory(config.MEMORY_DIR, config.TIMELINE_MAX_CHARS)
        speaker = Speaker(
            config.KOKORO_MODEL_PATH,
            config.KOKORO_VOICES_PATH,
            config.KOKORO_VOICE,
            config.KOKORO_SPEED,
        )
    except Exception as exc:
        emit({"event": "error", "text": str(exc)})
        _close_capture(capture)
        return False

    if ptt:
        emit(
            {
                "event": "system",
                "text": f"Ready. Hold your push-to-talk key and say "
                f'"{config.WAKE_PHRASE[0]}" to start.',
            }
        )
    else:
        emit({"event": "system", "text": f'Ready. Say "{config.WAKE_PHRASE[0]}" to start.'})

    try:
        run_loop(
            capture,
            transcriber,
            llm,
            memory,
            speaker,
            StateMachine(),
            emit,
            should_stop,
        )
    finally:
        _close_capture(capture)
    return True
```

Note: `run_session` intentionally does not catch `KeyboardInterrupt` — the terminal front-end handles it; the `finally` still closes the capture.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_session.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add companion/session.py tests/test_session.py
git commit -m "feat: session preflight checks and run_session orchestration"
```

---

### Task 4: Rewire `main.py` (terminal front-end) onto the session engine

`main.py` keeps its menus and flags but delegates everything after them to `session.run_session` with a print-based emitter. `remember_session` and the check helpers leave `main.py` (they now live in `session.py`).

**Files:**
- Modify: `companion/main.py` (full rewrite below)
- Modify: `tests/test_main.py`

**Interfaces:**
- Consumes: `session.run_session(brain, ears, ptt, emit, should_stop) -> bool`, `session.PROVIDER_NAMES`, `session.STT_NAMES`.
- Produces: `main.PROVIDER_NAMES` / `main.STT_NAMES` (re-exported so existing imports keep working), `choose_from_menu`, `ask_yes_no`, `print_event(event) -> None`.

- [ ] **Step 1: Update the tests**

Replace the imports and the `remember_session` tests in `tests/test_main.py`. The file becomes:

```python
# tests/test_main.py
from unittest.mock import patch

from companion.main import (
    PROVIDER_NAMES,
    STT_NAMES,
    ask_yes_no,
    choose_from_menu,
    print_event,
)


def test_cli_flag_skips_the_menu():
    with patch("builtins.input") as mock_input:
        assert choose_from_menu("brain", PROVIDER_NAMES, "local", "claude") == "claude"
    mock_input.assert_not_called()


def test_empty_input_returns_default():
    with patch("builtins.input", return_value=""):
        assert choose_from_menu("brain", PROVIDER_NAMES, "local", None) == "local"


def test_number_input_picks_from_menu():
    with patch("builtins.input", return_value="2"):
        assert choose_from_menu("brain", PROVIDER_NAMES, "local", None) == "claude"


def test_name_input_picks_choice():
    with patch("builtins.input", return_value="zai"):
        assert choose_from_menu("brain", PROVIDER_NAMES, "local", None) == "zai"


def test_garbage_input_falls_back_to_default():
    with patch("builtins.input", return_value="skynet"):
        assert choose_from_menu("brain", PROVIDER_NAMES, "local", None) == "local"


def test_ears_menu_number_picks_openai():
    with patch("builtins.input", return_value="2"):
        assert choose_from_menu("ears", STT_NAMES, "local", None) == "openai"


def test_ears_cli_flag_skips_menu():
    with patch("builtins.input") as mock_input:
        assert choose_from_menu("ears", STT_NAMES, "local", "openai") == "openai"
    mock_input.assert_not_called()


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


def test_print_event_formats_each_kind(capsys):
    print_event({"event": "heard", "text": "hi"})
    print_event({"event": "reply", "text": "hello"})
    print_event({"event": "system", "text": "Waking up."})
    print_event({"event": "warning", "text": "oops"})
    print_event({"event": "error", "text": "bad"})
    print_event({"event": "status", "state": "listening"})  # silent
    print_event({"event": "session_ended"})  # silent
    out = capsys.readouterr().out
    assert "Heard: hi" in out
    assert "Companion: hello" in out
    assert "Waking up." in out
    assert "WARNING: oops" in out
    assert "ERROR: bad" in out
    assert "listening" not in out
```

(The four `remember_session` tests are deleted here — Task 2 already ported them to `tests/test_session.py`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_main.py -q`
Expected: FAIL with `ImportError: cannot import name 'print_event'`.

- [ ] **Step 3: Rewrite `companion/main.py`**

Full new contents:

```python
# companion/main.py
import argparse
import sys

from dotenv import load_dotenv

from companion import config, session
from companion.session import PROVIDER_NAMES, STT_NAMES


def choose_from_menu(label, names, default, cli_choice) -> str:
    if cli_choice:
        return cli_choice
    print(f"Choose {label}:")
    for number, name in enumerate(names, start=1):
        marker = " (default)" if name == default else ""
        print(f"  {number}. {name}{marker}")
    answer = input("Number or name [Enter = default]: ").strip().lower()
    if not answer:
        return default
    if answer.isdigit() and 1 <= int(answer) <= len(names):
        return names[int(answer) - 1]
    if answer in names:
        return answer
    print(f"Unknown choice '{answer}', using {default}.")
    return default


def ask_yes_no(label, default, cli_flag) -> bool:
    if cli_flag:
        return True
    answer = input(f"{label} [y/N]: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def print_event(event) -> None:
    kind = event["event"]
    if kind == "heard":
        print(f"Heard: {event['text']}")
    elif kind == "reply":
        print(f"Companion: {event['text']}")
    elif kind == "system":
        print(event["text"])
    elif kind == "warning":
        print(f"WARNING: {event['text']}")
    elif kind == "error":
        print(f"ERROR: {event['text']}")
    # "status" and "session_ended" drive the web page; the terminal skips them.


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Voice English companion")
    parser.add_argument(
        "--brain",
        choices=PROVIDER_NAMES,
        default=None,
        help="skip the startup menu and use this provider",
    )
    parser.add_argument(
        "--ears",
        choices=STT_NAMES,
        default=None,
        help="skip the ears menu and use this transcription backend",
    )
    parser.add_argument(
        "--ptt",
        action="store_true",
        help="enable push-to-talk (hold PTT_KEY to record) and skip the prompt",
    )
    args = parser.parse_args()
    brain = choose_from_menu("brain", PROVIDER_NAMES, config.LLM_PROVIDER, args.brain)
    print(f"Brain: {brain}")
    ears = choose_from_menu(
        "ears (transcription)", STT_NAMES, config.STT_PROVIDER, args.ears
    )
    print(f"Ears: {ears}")
    ptt = ask_yes_no("Push-to-talk (hold a key to record)?", False, args.ptt)
    print(f"Push-to-talk: {'on' if ptt else 'off'}")

    print("Checking services and microphone...")
    try:
        ok = session.run_session(brain, ears, ptt, print_event, lambda: False)
    except KeyboardInterrupt:
        print("\nExiting.")
        return
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 5: Manual terminal smoke test**

Run: `python -m companion.main --brain local --ears local` (with Ollama running) or expect a clean `ERROR: Could not reach Ollama (...)` exit if not. Verify the menus/flags behave, say "hey chat", get a spoken greeting, say "bye bye", Ctrl+C exits cleanly. If no mic/Ollama available, at minimum verify the error paths print and exit(1).

- [ ] **Step 6: Commit**

```bash
git add companion/main.py tests/test_main.py
git commit -m "refactor: main.py delegates to the session engine"
```

---

### Task 5: FastAPI server — SessionManager, WebSocket, memory endpoint

The web backend. One session thread at a time; events fan out to all connected sockets and accumulate in a bounded history replayed on (re)connect. Includes a placeholder `companion/web/index.html` so the static mount works — Task 6 replaces it with the real UI.

**Files:**
- Modify: `requirements.txt`
- Create: `companion/server.py`
- Create: `companion/web/index.html` (placeholder)
- Create: `tests/test_server.py`

**Interfaces:**
- Consumes: `session.run_session`, `session.PROVIDER_NAMES`, `session.STT_NAMES`, `Memory` (for file paths), `config.MEMORY_DIR`, `config.TIMELINE_MAX_CHARS`, `config.LLM_PROVIDER`, `config.STT_PROVIDER`.
- Produces (used by Task 6's JS):
  - `GET /` — the static page.
  - `GET /api/options` → `{"brains": [...], "ears": [...], "defaults": {"brain": ..., "ears": ..., "ptt": false}}`
  - `GET /api/memory` → `{"durable": "<file text>", "timeline": "<file text>"}`
  - `WS /ws` — on connect sends `{"event": "hello", "running": bool, "state": str}` then replays history; accepts `{"cmd": "start", "brain", "ears", "ptt"}` and `{"cmd": "stop"}`.
  - `python -m companion.server` runs uvicorn on 127.0.0.1:8000 and opens the browser.

- [ ] **Step 1: Add dependencies and install**

Append to `requirements.txt`:

```
fastapi>=0.110,<1
uvicorn>=0.30,<1
```

Run: `pip install -r requirements.txt`
Expected: fastapi + uvicorn install without errors. (`httpx`, needed by FastAPI's TestClient, comes with them; if `python -c "import httpx"` fails, run `pip install httpx`.)

- [ ] **Step 2: Write the failing tests**

Create `tests/test_server.py`:

```python
# tests/test_server.py
import time

from fastapi.testclient import TestClient

from companion import config, server


def fake_run_session(brain, ears, ptt, emit, should_stop):
    emit({"event": "system", "text": f"fake session {brain}/{ears}/{ptt}"})
    for _ in range(500):
        if should_stop():
            return True
        time.sleep(0.01)
    return True


def fresh_client():
    # Each test gets its own manager so threads/history never leak between
    # tests. TestClient used as a context manager runs the lifespan, which
    # captures the event loop for cross-thread broadcasting.
    server.manager = server.SessionManager()
    return TestClient(server.app)


def test_serves_the_page():
    with fresh_client() as client:
        res = client.get("/")
    assert res.status_code == 200
    assert "Companion" in res.text


def test_memory_endpoint_returns_both_files(tmp_path, monkeypatch):
    (tmp_path / "durable.md").write_text("## Facts\n- Likes ramen.\n", encoding="utf-8")
    (tmp_path / "timeline.md").write_text("## 2026-07-08\n- Chatted.\n", encoding="utf-8")
    monkeypatch.setattr(config, "MEMORY_DIR", str(tmp_path))
    with fresh_client() as client:
        data = client.get("/api/memory").json()
    assert "Likes ramen" in data["durable"]
    assert "Chatted" in data["timeline"]


def test_memory_endpoint_tolerates_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_DIR", str(tmp_path / "nope"))
    with fresh_client() as client:
        data = client.get("/api/memory").json()
    assert data == {"durable": "", "timeline": ""}


def test_options_endpoint_lists_choices_and_defaults():
    with fresh_client() as client:
        data = client.get("/api/options").json()
    assert data["brains"] == ["local", "claude", "openai", "zai"]
    assert data["ears"] == ["local", "openai"]
    assert data["defaults"] == {
        "brain": config.LLM_PROVIDER,
        "ears": config.STT_PROVIDER,
        "ptt": False,
    }


def test_websocket_start_and_stop_cycle(monkeypatch):
    monkeypatch.setattr(server.session, "run_session", fake_run_session)
    with fresh_client() as client:
        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello == {"event": "hello", "running": False, "state": "idle"}
            ws.send_json({"cmd": "start", "brain": "local", "ears": "local", "ptt": False})
            assert ws.receive_json() == {
                "event": "system",
                "text": "fake session local/local/False",
            }
            ws.send_json({"cmd": "stop"})
            assert ws.receive_json() == {"event": "status", "state": "idle"}
            assert ws.receive_json() == {"event": "session_ended"}


def test_second_start_is_rejected_while_running(monkeypatch):
    monkeypatch.setattr(server.session, "run_session", fake_run_session)
    with fresh_client() as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # hello
            ws.send_json({"cmd": "start", "brain": "local", "ears": "local", "ptt": False})
            ws.receive_json()  # fake session line
            ws.send_json({"cmd": "start", "brain": "claude", "ears": "local", "ptt": False})
            assert ws.receive_json() == {
                "event": "error",
                "text": "A session is already running.",
            }
            ws.send_json({"cmd": "stop"})
            ws.receive_json()  # status idle
            ws.receive_json()  # session_ended


def test_start_with_unknown_option_is_rejected():
    with fresh_client() as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # hello
            ws.send_json({"cmd": "start", "brain": "skynet", "ears": "local", "ptt": False})
            event = ws.receive_json()
    assert event["event"] == "error"
    assert "skynet" in event["text"]


def test_reconnect_replays_history(monkeypatch):
    monkeypatch.setattr(server.session, "run_session", fake_run_session)
    with fresh_client() as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # hello
            ws.send_json({"cmd": "start", "brain": "local", "ears": "local", "ptt": False})
            ws.receive_json()  # fake session line
        # Simulated page refresh: hello reports running, history replays.
        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello["running"] is True
            assert ws.receive_json() == {
                "event": "system",
                "text": "fake session local/local/False",
            }
            ws.send_json({"cmd": "stop"})
            ws.receive_json()  # status idle
            ws.receive_json()  # session_ended
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_server.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'companion.server'`.

- [ ] **Step 4: Create the placeholder page and `companion/server.py`**

Create `companion/web/index.html` (placeholder — Task 6 replaces it):

```html
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Companion</title></head>
<body><h1>Companion</h1><p>Web UI coming in the next task.</p></body>
</html>
```

Create `companion/server.py`:

```python
# companion/server.py
"""Web front-end: serves the static page, fans session events out over a
WebSocket, and runs at most one voice session in a background thread. Run
with: python -m companion.server"""
import asyncio
import os
import threading
import webbrowser
from collections import deque
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from companion import config, session
from companion.memory import Memory

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
HOST = "127.0.0.1"
PORT = 8000
URL = f"http://localhost:{PORT}"


class SessionManager:
    """Owns the single session thread and the fan-out to websockets.

    emit() is called from the session thread; it hops onto the server's event
    loop via run_coroutine_threadsafe, so sockets are only touched from the
    loop. history is bounded and replayed to (re)connecting pages so a
    refresh shows the whole conversation so far."""

    def __init__(self):
        self.loop = None  # captured by the lifespan at startup
        self.sockets = []
        self.history = deque(maxlen=1000)
        self.state = "idle"
        self.thread = None
        self.stop_event = threading.Event()

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, brain, ears, ptt) -> None:
        if self.running:
            self.emit({"event": "error", "text": "A session is already running."})
            return
        if brain not in session.PROVIDER_NAMES or ears not in session.STT_NAMES:
            self.emit({"event": "error", "text": f"Unknown option: {brain}/{ears}."})
            return
        self.history.clear()
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run, args=(brain, ears, bool(ptt)), daemon=True
        )
        self.thread.start()

    def _run(self, brain, ears, ptt) -> None:
        session.run_session(brain, ears, ptt, self.emit, self.stop_event.is_set)
        # Always emitted, even when run_session failed preflight — this is
        # what unlocks the options panel in the browser.
        self.emit({"event": "status", "state": "idle"})
        self.emit({"event": "session_ended"})

    def stop(self) -> None:
        self.stop_event.set()

    def emit(self, event) -> None:
        if event["event"] == "status":
            self.state = event["state"]
        self.history.append(event)
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self._broadcast(event), self.loop)

    async def _broadcast(self, event) -> None:
        for ws in list(self.sockets):
            try:
                await ws.send_json(event)
            except Exception:
                # A socket that died mid-send; drop it, the page reconnects.
                if ws in self.sockets:
                    self.sockets.remove(ws)


manager = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.loop = asyncio.get_running_loop()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/api/options")
def get_options():
    return {
        "brains": session.PROVIDER_NAMES,
        "ears": session.STT_NAMES,
        "defaults": {
            "brain": config.LLM_PROVIDER,
            "ears": config.STT_PROVIDER,
            "ptt": False,
        },
    }


@app.get("/api/memory")
def get_memory():
    memory = Memory(config.MEMORY_DIR, config.TIMELINE_MAX_CHARS)
    timeline = ""
    if os.path.exists(memory.timeline_path):
        with open(memory.timeline_path, "r", encoding="utf-8") as f:
            timeline = f.read().strip()
    return {"durable": memory.load_durable(), "timeline": timeline}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    manager.sockets.append(ws)
    await ws.send_json(
        {"event": "hello", "running": manager.running, "state": manager.state}
    )
    for event in list(manager.history):
        await ws.send_json(event)
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("cmd") == "start":
                manager.start(msg.get("brain"), msg.get("ears"), msg.get("ptt"))
            elif msg.get("cmd") == "stop":
                manager.stop()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in manager.sockets:
            manager.sockets.remove(ws)


# Mounted last so /api and /ws win the route match.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def main() -> None:
    load_dotenv()
    print(f"Companion web interface: {URL}")
    # uvicorn.run blocks; open the browser shortly after it comes up.
    threading.Timer(1.0, webbrowser.open, args=[URL]).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_server.py -q`
Expected: all PASS. Then run the whole suite: `python -m pytest tests/ -q` — all PASS.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt companion/server.py companion/web/index.html tests/test_server.py
git commit -m "feat: FastAPI server with session manager, WebSocket, and memory API"
```

---

### Task 6: The web page — dark Claude-style chat UI

Replace the placeholder with the real interface: sidebar (session options + memory tabs), chat area with bubbles and status pill, WebSocket client with auto-reconnect, and a minimal markdown renderer for the memory files. No frameworks, no CDN.

**Files:**
- Modify: `companion/web/index.html` (full replacement)
- Create: `companion/web/style.css`
- Create: `companion/web/app.js`

**Interfaces:**
- Consumes: exactly the Task 5 HTTP/WS surface (`/api/options`, `/api/memory`, `/ws`, the `hello` + protocol events).
- Produces: the user-facing page. No JS is imported by other code.

- [ ] **Step 1: Replace `companion/web/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Companion</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <aside class="sidebar">
    <h1>Companion</h1>
    <section class="card">
      <h2>Session</h2>
      <label for="brain">Brain</label>
      <select id="brain"></select>
      <label for="ears">Ears</label>
      <select id="ears"></select>
      <label class="toggle">
        <input type="checkbox" id="ptt">
        <span>Push-to-talk</span>
      </label>
      <button id="session-btn">Start</button>
    </section>
    <section class="card">
      <h2>Memory</h2>
      <div class="tabs">
        <button class="tab active" data-tab="durable">About you</button>
        <button class="tab" data-tab="timeline">Timeline</button>
      </div>
      <div id="memory-content" class="memory-content"></div>
    </section>
  </aside>
  <main class="main">
    <header class="topbar">
      <span id="status-pill" class="pill idle">Idle</span>
      <span id="conn" class="conn"></span>
    </header>
    <div id="error-banner" class="banner hidden"></div>
    <div id="chat" class="chat"></div>
  </main>
  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `companion/web/style.css`**

```css
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: "Segoe UI", system-ui, sans-serif;
  background: #262624;
  color: #e8e6e1;
  display: flex;
  height: 100vh;
  overflow: hidden;
}

/* ---------- sidebar ---------- */
.sidebar {
  width: 300px;
  min-width: 300px;
  background: #1f1e1c;
  border-right: 1px solid #3a3833;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  overflow-y: auto;
}
.sidebar h1 { font-size: 20px; color: #d97757; }

.card {
  background: #262624;
  border: 1px solid #3a3833;
  border-radius: 12px;
  padding: 16px;
}
.card h2 {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #a8a49c;
  margin-bottom: 12px;
}

label { display: block; font-size: 13px; color: #a8a49c; margin: 8px 0 4px; }
select {
  width: 100%;
  padding: 8px;
  border-radius: 8px;
  border: 1px solid #3a3833;
  background: #1f1e1c;
  color: #e8e6e1;
  font-size: 14px;
}
select:disabled { opacity: 0.5; }

.toggle {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 12px 0;
  color: #e8e6e1;
  font-size: 14px;
}
.toggle input { accent-color: #d97757; width: 16px; height: 16px; }
.toggle input:disabled { opacity: 0.5; }

#session-btn {
  width: 100%;
  padding: 10px;
  margin-top: 8px;
  border: none;
  border-radius: 8px;
  background: #d97757;
  color: #1a1915;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
}
#session-btn:hover { filter: brightness(1.1); }
#session-btn.running { background: #8a3b2e; color: #f0ded9; }

.tabs { display: flex; gap: 4px; margin-bottom: 10px; }
.tab {
  flex: 1;
  padding: 6px;
  font-size: 12px;
  border: 1px solid #3a3833;
  border-radius: 6px;
  background: transparent;
  color: #a8a49c;
  cursor: pointer;
}
.tab.active { background: #3a3833; color: #e8e6e1; }

.memory-content {
  font-size: 13px;
  line-height: 1.5;
  color: #cfccc4;
  max-height: 40vh;
  overflow-y: auto;
}
.memory-content h3 { font-size: 13px; color: #d97757; margin: 10px 0 4px; }
.memory-content ul { padding-left: 18px; }
.memory-content p { margin: 4px 0; }
.memory-content .empty { color: #6f6b63; font-style: italic; }

/* ---------- main column ---------- */
.main { flex: 1; display: flex; flex-direction: column; min-width: 0; }

.topbar {
  padding: 14px 24px;
  border-bottom: 1px solid #3a3833;
  display: flex;
  align-items: center;
  gap: 12px;
}
.pill {
  padding: 4px 14px;
  border-radius: 999px;
  font-size: 13px;
  background: #3a3833;
  color: #a8a49c;
}
.pill.listening { background: #2e4a2e; color: #9fd39f; }
.pill.thinking  { background: #4a3d24; color: #e6c47a; }
.pill.speaking  { background: #4a3227; color: #e8a188; }
.pill.loading   { background: #3a3833; color: #e8e6e1; }
.conn { font-size: 12px; color: #e6c47a; }

.banner {
  margin: 12px 24px 0;
  padding: 12px 16px;
  border-radius: 8px;
  background: #4a2323;
  border: 1px solid #7a3b3b;
  color: #f0b9b9;
  font-size: 14px;
}
.hidden { display: none; }

/* ---------- chat ---------- */
.chat {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.row { display: flex; }
.row.user { justify-content: flex-end; }
.row.companion { justify-content: flex-start; }

.bubble {
  max-width: 65%;
  padding: 10px 14px;
  border-radius: 16px;
  font-size: 15px;
  line-height: 1.45;
  white-space: pre-wrap;
  overflow-wrap: break-word;
}
.row.user .bubble { background: #4a3227; border-bottom-right-radius: 4px; }
.row.companion .bubble { background: #33312c; border-bottom-left-radius: 4px; }
.stamp { font-size: 11px; color: #8f8b83; margin-top: 4px; text-align: right; }

.line { text-align: center; font-size: 12.5px; }
.line.system { color: #8f8b83; }
.line.warning { color: #e6c47a; }
```

- [ ] **Step 3: Create `companion/web/app.js`**

```js
// companion/web/app.js
const chat = document.getElementById("chat");
const pill = document.getElementById("status-pill");
const conn = document.getElementById("conn");
const banner = document.getElementById("error-banner");
const brainSel = document.getElementById("brain");
const earsSel = document.getElementById("ears");
const pttBox = document.getElementById("ptt");
const btn = document.getElementById("session-btn");
const memoryContent = document.getElementById("memory-content");

let ws = null;
let running = false;
let memory = { durable: "", timeline: "" };
let activeTab = "durable";

const STATUS_LABELS = {
  idle: "Idle",
  loading: "Loading models…",
  listening: "Listening 🎤",
  thinking: "Thinking…",
  speaking: "Speaking 🔊",
};

function setStatus(state) {
  pill.textContent = STATUS_LABELS[state] || state;
  pill.className = "pill " + state;
}

function setRunning(isRunning) {
  running = isRunning;
  btn.textContent = isRunning ? "End session" : "Start";
  btn.classList.toggle("running", isRunning);
  for (const el of [brainSel, earsSel, pttBox]) el.disabled = isRunning;
}

function timeNow() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function addBubble(role, text) {
  const row = document.createElement("div");
  row.className = "row " + role;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  const stamp = document.createElement("div");
  stamp.className = "stamp";
  stamp.textContent = timeNow();
  bubble.appendChild(stamp);
  row.appendChild(bubble);
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
}

function addLine(kind, text) {
  const div = document.createElement("div");
  div.className = "line " + kind;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function showError(text) {
  banner.textContent = text;
  banner.classList.remove("hidden");
}

function handleEvent(ev) {
  switch (ev.event) {
    case "hello":
      setRunning(ev.running);
      setStatus(ev.state);
      break;
    case "status":
      setStatus(ev.state);
      break;
    case "heard":
      addBubble("user", ev.text);
      break;
    case "reply":
      addBubble("companion", ev.text);
      break;
    case "system":
      addLine("system", ev.text);
      break;
    case "warning":
      addLine("warning", "⚠ " + ev.text);
      break;
    case "error":
      showError(ev.text);
      break;
    case "session_ended":
      setRunning(false);
      setStatus("idle");
      loadMemory();
      break;
  }
}

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => {
    conn.textContent = "";
    // The server replays the session history right after hello; start from
    // a clean slate so a reconnect doesn't duplicate bubbles.
    chat.innerHTML = "";
    banner.classList.add("hidden");
  };
  ws.onmessage = (msg) => handleEvent(JSON.parse(msg.data));
  ws.onclose = () => {
    conn.textContent = "reconnecting…";
    setTimeout(connect, 1500);
  };
}

btn.addEventListener("click", () => {
  banner.classList.add("hidden");
  if (running) {
    ws.send(JSON.stringify({ cmd: "stop" }));
  } else {
    setRunning(true); // session_ended always unlocks, even on start failure
    ws.send(JSON.stringify({
      cmd: "start",
      brain: brainSel.value,
      ears: earsSel.value,
      ptt: pttBox.checked,
    }));
  }
});

// Minimal markdown for the memory files: ## headings and "- " bullets only.
function renderMarkdown(text) {
  const esc = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  let html = "";
  let inList = false;
  for (const line of text.split(/\r?\n/)) {
    const t = line.trim();
    if (t.startsWith("- ")) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += "<li>" + esc(t.slice(2)) + "</li>";
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      if (t.startsWith("## ")) html += "<h3>" + esc(t.slice(3)) + "</h3>";
      else if (t.startsWith("# ")) html += "<h3>" + esc(t.slice(2)) + "</h3>";
      else if (t) html += "<p>" + esc(t) + "</p>";
    }
  }
  if (inList) html += "</ul>";
  return html;
}

function showTab(name) {
  activeTab = name;
  for (const tab of document.querySelectorAll(".tab")) {
    tab.classList.toggle("active", tab.dataset.tab === name);
  }
  const text = memory[name] || "";
  memoryContent.innerHTML = text
    ? renderMarkdown(text)
    : "<p class=\"empty\">Nothing here yet.</p>";
}

for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => showTab(tab.dataset.tab));
}

async function loadMemory() {
  const res = await fetch("/api/memory");
  memory = await res.json();
  showTab(activeTab);
}

async function loadOptions() {
  const res = await fetch("/api/options");
  const opts = await res.json();
  for (const name of opts.brains) brainSel.add(new Option(name, name));
  for (const name of opts.ears) earsSel.add(new Option(name, name));
  brainSel.value = opts.defaults.brain;
  earsSel.value = opts.defaults.ears;
  pttBox.checked = opts.defaults.ptt;
}

loadOptions();
loadMemory();
connect();
```

- [ ] **Step 4: Run the automated suite (server tests still pass with the new page)**

Run: `python -m pytest tests/ -q`
Expected: all PASS (`test_serves_the_page` still finds "Companion" in the new HTML).

- [ ] **Step 5: Manual smoke test**

1. Run: `python -m companion.server` — browser opens `http://localhost:8000`.
2. Verify: dark two-column layout; brain/ears dropdowns populated with defaults selected; memory tabs show the contents of `memory/durable.md` and `memory/timeline.md` formatted (headings orange, bullets indented).
3. Click **Start** with Ollama stopped → red banner "Could not reach Ollama…", options unlock again.
4. Start Ollama (or pick a cloud brain with a key in `.env`), click **Start** → status pill goes "Loading models…" then "Listening 🎤"; say "hey chat" → greeting appears as a left bubble and is spoken; chat back and forth → your words appear as right bubbles.
5. Refresh the page mid-session → conversation replays, options stay locked, pill correct.
6. Click **End session** → goodbye spoken, "Remembering this session..." line, options unlock, memory panel refreshes with the new timeline entry.
7. Ctrl+C in the terminal stops the server.

- [ ] **Step 6: Commit**

```bash
git add companion/web/index.html companion/web/style.css companion/web/app.js
git commit -m "feat: dark Claude-style web chat page"
```

---

### Task 7: README, full-suite verification, wrap-up

**Files:**
- Modify: `README.md`

**Interfaces:** none new.

- [ ] **Step 1: Document the web interface in the README**

Add this section to `README.md` after the existing usage/run section (adjust heading level to match the file):

```markdown
## Web interface

Instead of the terminal prompts, you can run the companion with a browser UI:

    python -m companion.server

Your browser opens http://localhost:8000 (localhost only). Pick the brain,
ears, and push-to-talk in the sidebar and click Start — the conversation
shows up as chat bubbles with a live status pill (listening / thinking /
speaking), and the sidebar shows what the companion remembers about you.
You still talk entirely by voice; End session (or saying "bye bye") saves
memory exactly like the terminal version. `python -m companion.main` keeps
working if you prefer the terminal.
```

- [ ] **Step 2: Run the full test suite one last time**

Run: `python -m pytest tests/ -q`
Expected: all PASS, no warnings about missing modules.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: web interface section in README"
```
