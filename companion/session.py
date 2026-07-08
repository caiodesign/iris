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
