# companion/main.py
import os
import sys

import numpy as np
import ollama
import sounddevice as sd

from companion import config
from companion.llm_client import LLMClient
from companion.speaker import Speaker
from companion.state_machine import Action, StateMachine
from companion.transcriber import Transcriber
from companion.voice_detector import VoiceDetector


def check_ollama_reachable() -> None:
    try:
        ollama.list()
    except Exception as exc:
        print(f"ERROR: Could not reach Ollama ({exc}). Is it running?")
        sys.exit(1)


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


def load_transcriber() -> Transcriber:
    try:
        transcriber = Transcriber(
            config.WHISPER_MODEL_SIZE, config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE
        )
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
    print("Checking Ollama and microphone...")
    check_ollama_reachable()
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
    transcriber = load_transcriber()
    llm = LLMClient(config.OLLAMA_MODEL, config.SYSTEM_PROMPT)
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
            text = transcriber.transcribe(audio)
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
                llm.reset()
                llm.seed_assistant(config.GREETING)
                speak_safely(speaker, config.GREETING)
            elif action == Action.CANCEL:
                print("Discarded that.")
            elif action == Action.SLEEP:
                print("Going back to sleep.")
                speak_safely(speaker, "Bye for now!")
            elif action == Action.FORWARD:
                reply = llm.send(text)
                print(f"Companion: {reply}")
                speak_safely(speaker, reply)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
