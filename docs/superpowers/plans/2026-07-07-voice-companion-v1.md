# English Voice Companion v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a locally-run, terminal-launched, voice-controlled English conversation companion: wakes on "Hey Chat," converses via a local LLM, speaks replies aloud, discards a misspoken utterance on "Cancel That," and returns to sleep on "Bye Bye."

**Architecture:** A blocking main loop wires five components together: a VAD-based `VoiceDetector` captures one spoken utterance at a time from the mic, a `Transcriber` (faster-whisper) converts it to text, a `StateMachine` decides whether that text is a control phrase or content to forward, an `LLMClient` (Ollama) generates replies for forwarded content, and a `Speaker` (Piper) speaks replies aloud. Each component is a small class with one job, independently unit-testable via mocking except where noted.

**Tech Stack:** Python 3.11+, faster-whisper (STT), Ollama Python client + `llama3.1:8b` (LLM), piper-tts (TTS), webrtcvad (installed as `webrtcvad-wheels`) + sounddevice (mic capture/VAD), pytest (tests).

## Global Constraints

- Control phrases (case- and punctuation-insensitive substring match — transcripts are lowercased and stripped of punctuation before matching, so Whisper's "Hey, Chat!" still matches), exactly as specified: wake = `"hey chat"`, cancel = `"cancel that"`, stop = `"bye bye"`.
- LLM: Ollama running model `llama3.1:8b`.
- Runtime: manually launched from a terminal window; no background/tray mode; exits when the window is closed.
- No pronunciation feedback, no barge-in/interrupt-while-speaking, no `memory.md` read/write in v1 (per spec, deferred).
- All processing local — no cloud API calls.

---

## Task 1: Project Scaffolding, Config, and Setup Docs

**Files:**
- Create: `requirements.txt`
- Create: `README.md`
- Create: `companion/__init__.py`
- Create: `companion/config.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: module `companion.config` with constants `WAKE_PHRASE`, `CANCEL_PHRASE`, `STOP_PHRASE`, `OLLAMA_MODEL`, `WHISPER_MODEL_SIZE`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`, `SAMPLE_RATE`, `FRAME_DURATION_MS`, `SILENCE_TIMEOUT_MS`, `PREROLL_MS`, `VAD_AGGRESSIVENESS`, `PIPER_VOICE_PATH`, `SYSTEM_PROMPT`, `GREETING` — consumed by every later task.

- [ ] **Step 1: Create `requirements.txt`**

```
faster-whisper>=1.0,<2
ollama>=0.4,<1
piper-tts>=1.3,<2
webrtcvad-wheels>=2.0,<3
sounddevice>=0.4,<1
numpy>=1.24,<3
pytest>=8,<9
```

(Major versions are pinned because the code samples in this plan target these APIs — piper-tts in particular changed its Python API completely between 0.x and 1.x. `webrtcvad-wheels` is the original `webrtcvad` library republished with prebuilt Windows/macOS/Linux wheels — plain `webrtcvad` has no Windows wheel and demands a C++ compiler to install. The import name is still `import webrtcvad`.)

- [ ] **Step 2: Create `companion/__init__.py` and `tests/__init__.py`** (both empty files, make each directory an importable package)

- [ ] **Step 3: Create `companion/config.py`**

```python
WAKE_PHRASE = "hey chat"
CANCEL_PHRASE = "cancel that"
STOP_PHRASE = "bye bye"

OLLAMA_MODEL = "llama3.1:8b"

WHISPER_MODEL_SIZE = "base.en"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
SILENCE_TIMEOUT_MS = 800
PREROLL_MS = 300
VAD_AGGRESSIVENESS = 2

PIPER_VOICE_PATH = "en_US-lessac-medium.onnx"

GREETING = "Hi! What would you like to work on today?"

SYSTEM_PROMPT = (
    "You are a friendly, encouraging English conversation companion helping "
    "the user practice English by voice. At the very start of a session, ask "
    "what they'd like to focus on today as an open question (for example: "
    "free conversation, grammar correction, or vocabulary building) rather "
    "than reading a fixed menu, then adapt your style to their answer. Keep "
    "replies conversational and reasonably short, since they will be spoken "
    "aloud, not read."
)
```

- [ ] **Step 4: Create `README.md` with setup instructions**

```markdown
# English Voice Companion

Local, voice-controlled English practice companion. Say "Hey Chat" to start
talking, "Cancel That" to retract what you just said, "Bye Bye" to end the
session.

## Setup

1. Install [Ollama](https://ollama.com) separately (not via pip), then pull the model:
   ```
   ollama pull llama3.1:8b
   ```
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install the NVIDIA libraries faster-whisper needs to run on the GPU
   (plain `pip install faster-whisper` does NOT include them):
   ```
   pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
   ```
   If the app still fails at startup with a CUDA/cuDNN error, open
   `companion/config.py` and set `WHISPER_DEVICE = "cpu"` and
   `WHISPER_COMPUTE_TYPE = "int8"` — slower, but always works.
4. Download a Piper voice (run from the project root, so the file lands here):
   ```
   python -m piper.download_voices en_US-lessac-medium
   ```
5. Run it:
   ```
   python -m companion.main
   ```

## Usage notes

- Say "Cancel That" **in the same breath** as the sentence you want to
  retract (e.g., "I went to... cancel that"). Once you pause, the app has
  already sent what you said to the model and a reply is on its way.
```

- [ ] **Step 5: Verify the package imports**

Run: `python -c "from companion import config; print(config.WAKE_PHRASE)"`
Expected output: `hey chat`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt README.md companion/__init__.py companion/config.py tests/__init__.py
git commit -m "chore: scaffold project, add config and setup docs"
```

---

## Task 2: State Machine

**Files:**
- Create: `companion/state_machine.py`
- Test: `tests/test_state_machine.py`

**Interfaces:**
- Consumes: `companion.config.WAKE_PHRASE`, `CANCEL_PHRASE`, `STOP_PHRASE` (strings).
- Produces: `State` enum (`ASLEEP`, `ACTIVE`), `Action` enum (`IGNORE`, `WAKE`, `SLEEP`, `CANCEL`, `FORWARD`), `StateMachine` class with `.state` attribute and `.process(text: str) -> Action` method — consumed by Task 7 (main loop).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_state_machine.py
from companion.state_machine import StateMachine, State, Action


def test_starts_asleep():
    machine = StateMachine()
    assert machine.state == State.ASLEEP


def test_asleep_ignores_unrelated_text():
    machine = StateMachine()
    action = machine.process("what a nice day today")
    assert action == Action.IGNORE
    assert machine.state == State.ASLEEP


def test_asleep_wakes_on_wake_phrase():
    machine = StateMachine()
    action = machine.process("Hey Chat, how are you")
    assert action == Action.WAKE
    assert machine.state == State.ACTIVE


def test_wake_detection_is_case_insensitive():
    machine = StateMachine()
    action = machine.process("HEY CHAT")
    assert action == Action.WAKE


def test_wake_detection_survives_whisper_punctuation():
    # Whisper routinely returns "Hey, Chat!" — a raw substring check on
    # "hey chat" would miss it because of the comma.
    machine = StateMachine()
    action = machine.process("Hey, Chat!")
    assert action == Action.WAKE
    assert machine.state == State.ACTIVE


def test_stop_detection_survives_hyphenation():
    # Whisper routinely returns "Bye-bye!" for a spoken "bye bye".
    machine = StateMachine()
    machine.process("hey chat")
    action = machine.process("Bye-bye!")
    assert action == Action.SLEEP
    assert machine.state == State.ASLEEP


def test_cancel_detection_survives_punctuation():
    machine = StateMachine()
    machine.process("hey chat")
    action = machine.process("I went to the store, cancel that!")
    assert action == Action.CANCEL


def test_active_forwards_normal_speech():
    machine = StateMachine()
    machine.process("hey chat")
    action = machine.process("I want to practice grammar today")
    assert action == Action.FORWARD
    assert machine.state == State.ACTIVE


def test_active_cancels_on_cancel_phrase():
    machine = StateMachine()
    machine.process("hey chat")
    action = machine.process("I want to talk about, cancel that")
    assert action == Action.CANCEL
    assert machine.state == State.ACTIVE


def test_active_sleeps_on_stop_phrase():
    machine = StateMachine()
    machine.process("hey chat")
    action = machine.process("okay bye bye")
    assert action == Action.SLEEP
    assert machine.state == State.ASLEEP


def test_returns_to_asleep_behavior_after_sleeping():
    machine = StateMachine()
    machine.process("hey chat")
    machine.process("bye bye")
    action = machine.process("random chatter")
    assert action == Action.IGNORE
    action = machine.process("hey chat again please")
    assert action == Action.WAKE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_state_machine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'companion.state_machine'`

- [ ] **Step 3: Write the implementation**

```python
# companion/state_machine.py
import re
from enum import Enum, auto

from companion.config import CANCEL_PHRASE, STOP_PHRASE, WAKE_PHRASE


class State(Enum):
    ASLEEP = auto()
    ACTIVE = auto()


class Action(Enum):
    IGNORE = auto()
    WAKE = auto()
    SLEEP = auto()
    CANCEL = auto()
    FORWARD = auto()


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation so Whisper's "Hey, Chat!" or
    "Bye-bye!" still match the plain control phrases."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class StateMachine:
    def __init__(self):
        self.state = State.ASLEEP

    def process(self, text: str) -> Action:
        normalized = _normalize(text)

        if self.state == State.ASLEEP:
            if WAKE_PHRASE in normalized:
                self.state = State.ACTIVE
                return Action.WAKE
            return Action.IGNORE

        if STOP_PHRASE in normalized:
            self.state = State.ASLEEP
            return Action.SLEEP
        if CANCEL_PHRASE in normalized:
            return Action.CANCEL
        return Action.FORWARD
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_state_machine.py -v`
Expected: PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add companion/state_machine.py tests/test_state_machine.py
git commit -m "feat: add conversation state machine with wake/cancel/stop phrases"
```

---

## Task 3: Transcriber (Speech-to-Text wrapper)

**Files:**
- Create: `companion/transcriber.py`
- Test: `tests/test_transcriber.py`

**Interfaces:**
- Consumes: `companion.config.WHISPER_MODEL_SIZE`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`.
- Produces: `Transcriber` class with `__init__(self, model_size: str, device: str, compute_type: str)` and `.transcribe(audio: np.ndarray) -> str` — consumed by Task 7 (main loop).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transcriber.py
from unittest.mock import MagicMock, patch

import numpy as np

from companion.transcriber import Transcriber


def test_transcribe_joins_and_strips_segment_text():
    fake_segment_1 = MagicMock(text=" Hello ")
    fake_segment_2 = MagicMock(text=" world ")

    with patch("companion.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (
            [fake_segment_1, fake_segment_2],
            None,
        )
        transcriber = Transcriber("base.en", "cpu", "int8")
        result = transcriber.transcribe(np.zeros(16000, dtype=np.float32))

    assert result == "Hello world"
    MockModel.assert_called_once_with("base.en", device="cpu", compute_type="int8")


def test_transcribe_returns_empty_string_for_silence():
    with patch("companion.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = ([], None)
        transcriber = Transcriber("base.en", "cpu", "int8")
        result = transcriber.transcribe(np.zeros(16000, dtype=np.float32))

    assert result == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_transcriber.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'companion.transcriber'`

- [ ] **Step 3: Write the implementation**

```python
# companion/transcriber.py
import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_size: str, device: str, compute_type: str):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, beam_size=5)
        return " ".join(segment.text.strip() for segment in segments).strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_transcriber.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add companion/transcriber.py tests/test_transcriber.py
git commit -m "feat: add faster-whisper transcriber wrapper"
```

---

## Task 4: LLM Client (Ollama wrapper)

**Files:**
- Create: `companion/llm_client.py`
- Test: `tests/test_llm_client.py`

**Interfaces:**
- Consumes: `companion.config.OLLAMA_MODEL`, `SYSTEM_PROMPT`.
- Produces: `LLMClient` class with `__init__(self, model: str, system_prompt: str)`, `.send(user_text: str) -> str`, `.reset(self) -> None`, `.seed_assistant(text: str) -> None` — consumed by Task 7 (main loop). `seed_assistant` exists because the app speaks a hardcoded greeting on wake; without recording that greeting in the history as an assistant turn, the LLM doesn't know it was said and would redundantly re-ask the same opening question.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_llm_client.py
from unittest.mock import patch

from companion.llm_client import LLMClient


def test_send_returns_reply_and_calls_ollama_with_history():
    fake_response = {"message": {"role": "assistant", "content": "Hi there!"}}
    with patch("companion.llm_client.ollama.chat", return_value=fake_response) as mock_chat:
        client = LLMClient("llama3.1:8b", "You are a helpful tutor.")
        reply = client.send("Hello")

    assert reply == "Hi there!"
    mock_chat.assert_called_once_with(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": "You are a helpful tutor."},
            {"role": "user", "content": "Hello"},
        ],
    )


def test_send_accumulates_conversation_history():
    fake_response = {"message": {"role": "assistant", "content": "Sure!"}}
    with patch("companion.llm_client.ollama.chat", return_value=fake_response) as mock_chat:
        client = LLMClient("llama3.1:8b", "system prompt")
        client.send("first message")
        client.send("second message")

    second_call_messages = mock_chat.call_args.kwargs["messages"]
    assert len(second_call_messages) == 4
    assert second_call_messages[-1] == {"role": "user", "content": "second message"}


def test_reset_clears_history_back_to_system_prompt():
    fake_response = {"message": {"role": "assistant", "content": "Sure!"}}
    with patch("companion.llm_client.ollama.chat", return_value=fake_response):
        client = LLMClient("llama3.1:8b", "system prompt")
        client.send("first message")
        client.reset()

    assert client.history == [{"role": "system", "content": "system prompt"}]


def test_seed_assistant_records_greeting_without_calling_ollama():
    client = LLMClient("llama3.1:8b", "system prompt")
    client.seed_assistant("Hi! What would you like to work on today?")

    assert client.history == [
        {"role": "system", "content": "system prompt"},
        {
            "role": "assistant",
            "content": "Hi! What would you like to work on today?",
        },
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'companion.llm_client'`

- [ ] **Step 3: Write the implementation**

```python
# companion/llm_client.py
import ollama


class LLMClient:
    def __init__(self, model: str, system_prompt: str):
        self.model = model
        self.system_prompt = system_prompt
        self.history = [{"role": "system", "content": system_prompt}]

    def send(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        response = ollama.chat(model=self.model, messages=self.history)
        reply = response["message"]["content"]
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        self.history = [{"role": "system", "content": self.system_prompt}]

    def seed_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_client.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add companion/llm_client.py tests/test_llm_client.py
git commit -m "feat: add Ollama LLM client wrapper with conversation history"
```

---

## Task 5: Speaker (Text-to-Speech wrapper)

**Files:**
- Create: `companion/speaker.py`
- Test: `tests/test_speaker.py`

**Interfaces:**
- Consumes: `companion.config.PIPER_VOICE_PATH`.
- Produces: `Speaker` class with `__init__(self, voice_path: str)` and `.speak(text: str) -> None` — consumed by Task 7 (main loop).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_speaker.py
from unittest.mock import patch

import numpy as np

from companion.speaker import Speaker


def fake_synthesize_wav(text, wav_file):
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(22050)
    wav_file.writeframes(np.array([100, -100, 200], dtype=np.int16).tobytes())


def test_speak_plays_synthesized_audio_at_voice_sample_rate():
    with patch("companion.speaker.PiperVoice") as MockVoice, patch(
        "companion.speaker.sd"
    ) as mock_sd:
        MockVoice.load.return_value.synthesize_wav.side_effect = fake_synthesize_wav

        speaker = Speaker("fake_voice.onnx")
        speaker.speak("Hello there")

    played_audio_arg = mock_sd.play.call_args.args[0]
    np.testing.assert_array_equal(
        played_audio_arg, np.array([100, -100, 200], dtype=np.int16)
    )
    assert mock_sd.play.call_args.kwargs["samplerate"] == 22050
    mock_sd.wait.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_speaker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'companion.speaker'`

- [ ] **Step 3: Write the implementation**

```python
# companion/speaker.py
import io
import wave

import numpy as np
import sounddevice as sd
from piper import PiperVoice


class Speaker:
    def __init__(self, voice_path: str):
        self.voice = PiperVoice.load(voice_path)

    def speak(self, text: str) -> None:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file)

        buffer.seek(0)
        with wave.open(buffer, "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_rate = wav_file.getframerate()

        audio = np.frombuffer(frames, dtype=np.int16)
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_speaker.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add companion/speaker.py tests/test_speaker.py
git commit -m "feat: add Piper text-to-speech speaker wrapper"
```

---

## Task 6: Voice Detector (mic capture + VAD)

**Files:**
- Create: `companion/voice_detector.py`

**Interfaces:**
- Consumes: `companion.config.SAMPLE_RATE`, `FRAME_DURATION_MS`, `SILENCE_TIMEOUT_MS`, `PREROLL_MS`, `VAD_AGGRESSIVENESS`.
- Produces: `VoiceDetector` class with `__init__(self, sample_rate, frame_duration_ms, silence_timeout_ms, preroll_ms, vad_aggressiveness)` and `.listen_for_utterance(self) -> np.ndarray` (blocks until one utterance is captured, returns mono float32 audio normalized to [-1, 1]) — consumed by Task 7 (main loop). The pre-roll ring buffer keeps the last `preroll_ms` of audio from *before* the VAD triggers and prepends it to the captured utterance — without it, the first syllable gets clipped and "hey chat" arrives at Whisper as "ey chat", which breaks wake detection.

**No automated test for this task.** This class does real microphone I/O via `sounddevice.InputStream`, which requires physical audio hardware and produces no meaningful assertion when mocked beyond "the mock was called" — the spec's own Testing Approach section already designates the full audio pipeline for manual verification rather than unit tests. This task is verified manually in Task 7's end-to-end check instead.

- [ ] **Step 1: Write the implementation**

```python
# companion/voice_detector.py
import collections

import numpy as np
import sounddevice as sd
import webrtcvad


class VoiceDetector:
    def __init__(
        self,
        sample_rate: int,
        frame_duration_ms: int,
        silence_timeout_ms: int,
        preroll_ms: int,
        vad_aggressiveness: int,
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.silence_timeout_frames = silence_timeout_ms // frame_duration_ms
        self.preroll_frames = preroll_ms // frame_duration_ms
        self.vad = webrtcvad.Vad(vad_aggressiveness)

    def listen_for_utterance(self) -> np.ndarray:
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

- [ ] **Step 2: Verify it imports cleanly**

Run: `python -c "from companion.voice_detector import VoiceDetector; print('ok')"`
Expected output: `ok`

- [ ] **Step 3: Commit**

```bash
git add companion/voice_detector.py
git commit -m "feat: add VAD-based voice detector for mic capture"
```

---

## Task 7: Main Loop, Startup Checks, and End-to-End Verification

**Files:**
- Create: `companion/main.py`

**Interfaces:**
- Consumes: `StateMachine`/`Action` (Task 2), `Transcriber` (Task 3), `LLMClient` (Task 4), `Speaker` (Task 5), `VoiceDetector` (Task 6), all of `companion.config`.
- Produces: `main()` entry point, runnable via `python -m companion.main`.

- [ ] **Step 1: Write the implementation**

```python
# companion/main.py
import sys

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


def load_transcriber() -> Transcriber:
    try:
        return Transcriber(
            config.WHISPER_MODEL_SIZE, config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE
        )
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
    speaker = Speaker(config.PIPER_VOICE_PATH)
    machine = StateMachine()

    print(f'Ready. Say "{config.WAKE_PHRASE}" to start.')

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
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `python -c "from companion.main import main; print('ok')"`
Expected output: `ok`

- [ ] **Step 3: Run the full automated test suite**

Run: `pytest -v`
Expected: all tests from Tasks 2-5 pass, no failures

- [ ] **Step 4: Commit**

```bash
git add companion/main.py
git commit -m "feat: wire components into main loop with startup checks"
```

- [ ] **Step 5: Manual end-to-end verification**

With Ollama running (`ollama pull llama3.1:8b` already done) and a Piper voice downloaded per the README, run:

```
python -m companion.main
```

Walk through the spec's v1 success criteria by voice:
1. Say "Hey Chat" → app should greet you and ask what you want to focus on.
2. Answer naturally (e.g., "let's just chat") → app should hold a normal spoken conversation.
3. Say a sentence ending in "cancel that" **in the same breath** (e.g., "I went to the store, cancel that") → app should discard it (no spoken reply, log shows "Discarded that.") and keep listening. Note: a standalone "cancel that" said *after* a pause cannot retract the previous utterance — that one was already sent to the LLM the moment you paused. This is expected v1 behavior, matching the spec.
4. Say "Bye Bye" → app should say goodbye and stop responding to further speech until "Hey Chat" is said again.
5. Confirm closing the terminal window fully exits the app.

If wake/stop/cancel detection misfires often during normal conversation, note it — that's the documented Option A latency/reliability tradeoff from the spec, and a candidate for revisiting with Option B (dedicated wake-word engine) later.
