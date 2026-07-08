# Voice + Personality Upgrade (v1.1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the robotic Piper TTS voice with Kokoro (`am_michael`) and give the companion a warm, charismatic personality via the system prompt.

**Architecture:** `Speaker` is rewritten around `kokoro-onnx` (CPU inference via onnxruntime — deliberately keeps VRAM free for Whisper and llama3.1) but keeps its public shape, so `main.py` only changes which config constants it passes. The personality change is config-only (`SYSTEM_PROMPT`, `GREETING`).

**Tech Stack:** Python 3.13 on Windows 11, kokoro-onnx, sounddevice, pytest. Spec: `docs/superpowers/specs/2026-07-08-voice-personality-upgrade-design.md`.

## Global Constraints

- Run tests with `python -m pytest` (bare `pytest` is not on PATH on this machine).
- All commands run from the repo root; model files land at the repo root.
- `Speaker.speak(text)` stays blocking (plays audio, waits for completion).
- Kokoro voice is `am_michael`, speed `1.0`, lang `"en-us"` (user's choice; must stay configurable via `companion/config.py`).
- All 21 existing tests must keep passing after every task.

---

### Task 1: Kokoro-based Speaker

**Files:**
- Modify: `requirements.txt` (swap `piper-tts` for `kokoro-onnx`)
- Modify: `companion/config.py` (add Kokoro constants; keep `PIPER_VOICE_PATH` for now — Task 2 removes it)
- Modify: `companion/speaker.py` (full rewrite)
- Modify: `.gitignore` (add `voices-v1.0.bin`)
- Test: `tests/test_speaker.py` (full rewrite)

**Interfaces:**
- Produces: `Speaker(model_path: str, voices_path: str, voice: str, speed: float)` with method `speak(text: str) -> None` (blocking). Config constants `KOKORO_MODEL_PATH`, `KOKORO_VOICES_PATH`, `KOKORO_VOICE`, `KOKORO_SPEED`.

- [ ] **Step 1: Swap the dependency and download model files**

In `requirements.txt`, replace the line `piper-tts>=1.3,<2` with:

```
kokoro-onnx>=0.4,<1
```

Then install and download the two model files (~330 MB total) to the repo root:

```bash
pip install "kokoro-onnx>=0.4,<1"
curl -L -o kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Add `voices-v1.0.bin` on its own line to `.gitignore` (`*.onnx` already covers the model file).

- [ ] **Step 2: Add Kokoro constants to config**

In `companion/config.py`, directly below the `PIPER_VOICE_PATH` line, add:

```python
KOKORO_MODEL_PATH = "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = "voices-v1.0.bin"
KOKORO_VOICE = "am_michael"
KOKORO_SPEED = 1.0
```

Do NOT remove `PIPER_VOICE_PATH` yet — `main.py` still references it until Task 2.

- [ ] **Step 3: Write the failing test**

Replace the entire contents of `tests/test_speaker.py` with:

```python
# tests/test_speaker.py
from unittest.mock import patch

import numpy as np

from companion.speaker import Speaker


def test_speak_plays_kokoro_audio_at_returned_sample_rate():
    fake_samples = np.array([0.1, -0.2, 0.3], dtype=np.float32)

    with patch("companion.speaker.Kokoro") as MockKokoro, patch(
        "companion.speaker.sd"
    ) as mock_sd:
        MockKokoro.return_value.create.return_value = (fake_samples, 24000)

        speaker = Speaker("fake_model.onnx", "fake_voices.bin", "am_michael", 1.0)
        speaker.speak("Hello there")

    MockKokoro.assert_called_once_with("fake_model.onnx", "fake_voices.bin")
    MockKokoro.return_value.create.assert_called_once_with(
        "Hello there", voice="am_michael", speed=1.0, lang="en-us"
    )
    played_audio_arg = mock_sd.play.call_args.args[0]
    np.testing.assert_array_equal(played_audio_arg, fake_samples)
    assert mock_sd.play.call_args.kwargs["samplerate"] == 24000
    mock_sd.wait.assert_called_once()
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_speaker.py -v`
Expected: FAIL with `ImportError` / `AttributeError` (no `Kokoro` in `companion.speaker`).

- [ ] **Step 5: Rewrite Speaker**

Replace the entire contents of `companion/speaker.py` with:

```python
# companion/speaker.py
import sounddevice as sd
from kokoro_onnx import Kokoro


class Speaker:
    def __init__(self, model_path: str, voices_path: str, voice: str, speed: float):
        self.kokoro = Kokoro(model_path, voices_path)
        self.voice = voice
        self.speed = speed

    def speak(self, text: str) -> None:
        samples, sample_rate = self.kokoro.create(
            text, voice=self.voice, speed=self.speed, lang="en-us"
        )
        sd.play(samples, samplerate=sample_rate)
        sd.wait()
```

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest -q`
Expected: 21 passed.

- [ ] **Step 7: Real synthesis smoke check (no playback)**

Run from the repo root:

```bash
python -c "from kokoro_onnx import Kokoro; k = Kokoro('kokoro-v1.0.onnx', 'voices-v1.0.bin'); s, sr = k.create('Testing the new voice.', voice='am_michael', speed=1.0, lang='en-us'); print('OK', len(s), sr)"
```

Expected: prints `OK <positive number> 24000` (sample rate may differ; must be > 0 samples, no exception).

- [ ] **Step 8: Commit**

```bash
git add requirements.txt companion/config.py companion/speaker.py tests/test_speaker.py .gitignore
git commit -m "feat: replace Piper with Kokoro TTS in Speaker"
```

---

### Task 2: Wire Kokoro into main, retire Piper

**Files:**
- Modify: `companion/main.py` (startup check + Speaker construction)
- Modify: `companion/config.py` (remove `PIPER_VOICE_PATH`)
- Modify: `README.md` (setup step 4)

**Interfaces:**
- Consumes: `Speaker(model_path, voices_path, voice, speed)` and config constants `KOKORO_MODEL_PATH`, `KOKORO_VOICES_PATH`, `KOKORO_VOICE`, `KOKORO_SPEED` from Task 1.

- [ ] **Step 1: Replace the voice-file startup check**

In `companion/main.py`, replace the whole `check_voice_file_available` function with:

```python
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
```

In `main()`, replace the call `check_voice_file_available()` with `check_tts_files_available()`.

- [ ] **Step 2: Construct Speaker with Kokoro config**

In `main()`, replace `speaker = Speaker(config.PIPER_VOICE_PATH)` with:

```python
speaker = Speaker(
    config.KOKORO_MODEL_PATH,
    config.KOKORO_VOICES_PATH,
    config.KOKORO_VOICE,
    config.KOKORO_SPEED,
)
```

- [ ] **Step 3: Remove PIPER_VOICE_PATH**

In `companion/config.py`, delete the line `PIPER_VOICE_PATH = "en_US-lessac-medium.onnx"`.

- [ ] **Step 4: Update README step 4**

In `README.md`, replace step 4 (the Piper voice download) with:

````markdown
4. Download the Kokoro voice model (two files, ~330 MB total, run from the
   project root so they land here):
   ```
   curl -L -o kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
   curl -L -o voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
   ```
   To change the voice or speaking speed, edit `KOKORO_VOICE` / `KOKORO_SPEED`
   in `companion/config.py` (voice list: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).
````

- [ ] **Step 5: Verify**

Run: `python -m pytest -q` — expected: 21 passed.
Run: `python -c "import companion.main"` — expected: no output, exit 0.
Check: `grep -rn PIPER companion/ tests/ README.md` returns nothing.

- [ ] **Step 6: Commit**

```bash
git add companion/main.py companion/config.py README.md
git commit -m "feat: wire Kokoro TTS into main and retire Piper"
```

---

### Task 3: Charismatic personality

**Files:**
- Modify: `companion/config.py` (`GREETING`, `SYSTEM_PROMPT`)

**Interfaces:**
- Consumes: nothing new. `GREETING` is still seeded as the first assistant turn by `main.py`; `SYSTEM_PROMPT` is still passed to `LLMClient`.

- [ ] **Step 1: Rewrite GREETING and SYSTEM_PROMPT**

In `companion/config.py`, replace the `GREETING` and `SYSTEM_PROMPT` assignments with:

```python
GREETING = "Hey, good to hear you! So — what are we diving into today?"

SYSTEM_PROMPT = (
    "You are Chat, a voice companion who helps the user practice English "
    "conversation. You have a real personality: warm, playful, and "
    "genuinely curious about the user's life. React to what they say the "
    "way a good friend would — surprise, delight, a little gentle teasing "
    "— instead of just answering. Ask follow-up questions about things "
    "they mention. Use light humor when it fits, and give encouragement "
    "only when they've earned it, so it means something. At the very "
    "start of a session, ask what they'd like to focus on today as an "
    "open question (for example: free conversation, grammar correction, "
    "or vocabulary building) rather than reading a fixed menu, then adapt "
    "your style to their answer. Your replies are spoken aloud, not read: "
    "keep them short (one to three sentences), natural, and free of "
    "lists, markdown, emojis, and stage directions."
)
```

- [ ] **Step 2: Verify**

Run: `python -m pytest -q` — expected: 21 passed (no test asserts on the exact prompt/greeting text; the greeting-seeding test uses `config.GREETING` indirectly through `main.py`, which is unaffected by the new value).

- [ ] **Step 3: Commit**

```bash
git add companion/config.py
git commit -m "feat: give the companion a warm, playful personality"
```

---

## Final verification (after Task 3)

1. `python -m pytest -q` → 21 passed.
2. Manual: `python -m companion.main`, say "Hey Chat" — confirm the greeting
   is spoken in the new `am_michael` voice, hold a short conversation, and
   judge warmth/personality. This is the user's acceptance test.
