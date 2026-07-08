# companion/main.py
import argparse
import os
import sys

import numpy as np
import ollama
import sounddevice as sd
from dotenv import load_dotenv

from companion import config
from companion.llm_client import LLMClient
from companion.memory import Memory
from companion.providers import REQUIRED_ENV, make_provider
from companion.speaker import Speaker
from companion.state_machine import Action, StateMachine
from companion.transcriber import make_transcriber
from companion.voice_detector import VoiceDetector

PROVIDER_NAMES = ["local", "claude", "openai", "zai"]
STT_NAMES = ["local", "openai"]


def check_ollama_reachable() -> None:
    try:
        ollama.list()
    except Exception as exc:
        print(f"ERROR: Could not reach Ollama ({exc}). Is it running?")
        sys.exit(1)


def check_api_key_available(brain: str) -> None:
    env_var = REQUIRED_ENV[brain]
    if not os.environ.get(env_var):
        print(
            f"ERROR: {env_var} is not set. Put it in a .env file at the "
            'project root (see the README section "Cloud brains") and try again.'
        )
        sys.exit(1)


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


def check_microphone_available() -> None:
    try:
        with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1, dtype="int16"):
            pass
    except Exception as exc:
        print(f"ERROR: Could not access a microphone ({exc}).")
        sys.exit(1)


def check_tts_files_available() -> None:
    missing = [
        path
        for path in (config.KOKORO_MODEL_PATH, config.KOKORO_VOICES_PATH)
        if not os.path.exists(path)
    ]
    if missing:
        print(
            f"ERROR: Kokoro TTS file(s) not found: {', '.join(missing)}. "
            "Run README step 4 from the project root to download them."
        )
        sys.exit(1)


def load_transcriber(ears: str):
    if ears == "openai":
        # Cloud ears: no GPU model to load, so no CUDA warm-up. The key was
        # already checked at startup.
        return make_transcriber("openai")
    try:
        transcriber = make_transcriber("local")
        # CUDA libraries load lazily on the first transcription, not at model
        # construction — warm up now so GPU problems surface here, with the
        # hint below, instead of as a traceback mid-conversation.
        transcriber.transcribe(np.zeros(config.SAMPLE_RATE, dtype=np.float32))
        return transcriber
    except Exception as exc:
        print(
            f"ERROR: Could not load the Whisper model on "
            f"'{config.WHISPER_DEVICE}' ({exc})."
        )
        print(
            "Hint: GPU mode needs NVIDIA libraries (see README step 3). To run "
            'on CPU instead, set WHISPER_DEVICE = "cpu" and '
            'WHISPER_COMPUTE_TYPE = "int8" in companion/config.py.'
        )
        sys.exit(1)


def speak_safely(speaker: Speaker, text: str) -> None:
    try:
        speaker.speak(text)
    except Exception as exc:
        print(f"WARNING: Text-to-speech playback failed ({exc}). Continuing without audio.")


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
    args = parser.parse_args()
    brain = choose_from_menu("brain", PROVIDER_NAMES, config.LLM_PROVIDER, args.brain)
    print(f"Brain: {brain}")
    ears = choose_from_menu(
        "ears (transcription)", STT_NAMES, config.STT_PROVIDER, args.ears
    )
    print(f"Ears: {ears}")

    print("Checking services and microphone...")
    if brain == "local":
        # Cloud brains never touch Ollama, so the llama model stays out of
        # VRAM — that's the point of using them while gaming.
        check_ollama_reachable()
    else:
        check_api_key_available(brain)
    if ears == "openai":
        # OpenAI cloud ears reuse the brain's OPENAI_API_KEY; fail fast if
        # absent (REQUIRED_ENV["openai"] == "OPENAI_API_KEY").
        check_api_key_available("openai")
    check_microphone_available()
    check_tts_files_available()

    print("Loading models (this may take a moment)...")
    detector = VoiceDetector(
        sample_rate=config.SAMPLE_RATE,
        frame_duration_ms=config.FRAME_DURATION_MS,
        silence_timeout_ms=config.SILENCE_TIMEOUT_MS,
        preroll_ms=config.PREROLL_MS,
        vad_aggressiveness=config.VAD_AGGRESSIVENESS,
    )
    transcriber = load_transcriber(ears)
    llm = LLMClient(make_provider(brain), config.SYSTEM_PROMPT)
    memory = Memory(config.MEMORY_PATH, config.MEMORY_MAX_CHARS)
    speaker = Speaker(
        config.KOKORO_MODEL_PATH,
        config.KOKORO_VOICES_PATH,
        config.KOKORO_VOICE,
        config.KOKORO_SPEED,
    )
    machine = StateMachine()

    print(f'Ready. Say "{config.WAKE_PHRASE[0]}" to start.')

    try:
        while True:
            audio = detector.listen_for_utterance()
            try:
                text = transcriber.transcribe(audio)
            except Exception as exc:
                # Cloud STT can fail on a network blip; a local frame can be
                # bad. Drop this utterance and keep listening instead of
                # crashing the session (mirrors the llm.send guard below).
                print(f"WARNING: Transcription failed ({exc}).")
                continue
            if not text:
                continue

            print(f"Heard: {text}")
            action = machine.process(text)

            if action == Action.IGNORE:
                continue
            elif action == Action.WAKE:
                print("Waking up.")
                # Fresh history per session, and the greeting is seeded as an
                # assistant turn — otherwise the LLM doesn't know the opening
                # question was already asked and re-asks it.
                llm.reset(memory.load())
                llm.seed_assistant(config.GREETING)
                speak_safely(speaker, config.GREETING)
            elif action == Action.CANCEL:
                print("Discarded that.")
            elif action == Action.SLEEP:
                print("Going back to sleep.")
                speak_safely(speaker, "Bye for now!")
                if llm.has_user_turns():
                    print("Remembering this session...")
                    try:
                        memory.append_session(llm.summarize(config.SUMMARY_PROMPT))
                    except Exception as exc:
                        print(f"WARNING: Could not save session memory ({exc}).")
            elif action == Action.FORWARD:
                try:
                    reply = llm.send(text)
                except Exception as exc:
                    # Cloud APIs hiccup and Ollama can die mid-session; keep
                    # the session alive. (The dangling user turn is harmless:
                    # every provider accepts consecutive user messages.)
                    print(f"WARNING: The brain failed to reply ({exc}).")
                    speak_safely(
                        speaker, "Sorry, I had trouble thinking. Say that again?"
                    )
                    continue
                print(f"Companion: {reply}")
                speak_safely(speaker, reply)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
