# companion/session.py
"""Session engine: the listen->transcribe->think->speak loop, decoupled from
any display. Callers pass emit(event_dict) and should_stop() and render the
events however they like — the terminal prints them, the web page streams
them over a WebSocket. Event shapes are the WebSocket protocol documented in
docs/superpowers/specs/2026-07-08-web-interface-design.md."""

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

PROVIDER_NAMES = ["local", "claude", "openai", "zai"]
STT_NAMES = ["local", "openai"]


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
