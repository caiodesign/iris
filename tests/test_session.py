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


def test_run_session_emits_error_when_loop_raises(monkeypatch):
    events = []
    closed = {"done": False}

    class DummyCapture:
        def close(self):
            closed["done"] = True

    monkeypatch.setattr(session, "preflight_error", lambda b, e: None)
    monkeypatch.setattr(session, "build_capture", lambda ptt: DummyCapture())
    monkeypatch.setattr(session, "load_transcriber", lambda ears: object())
    monkeypatch.setattr(session, "make_provider", lambda brain: object())
    monkeypatch.setattr(session, "Speaker", lambda *args: FakeSpeaker())

    def exploding_loop(*args, **kwargs):
        raise RuntimeError("mic exploded")

    monkeypatch.setattr(session, "run_loop", exploding_loop)
    ok = session.run_session("local", "local", False, events.append, lambda: True)
    assert ok is False
    errors = [e["text"] for e in events if e["event"] == "error"]
    assert any("mic exploded" in t for t in errors)
    assert closed["done"] is True
