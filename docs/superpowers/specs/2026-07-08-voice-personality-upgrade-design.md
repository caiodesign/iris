# Voice + Personality Upgrade (v1.1) — Design

**Date:** 2026-07-08
**Status:** Approved by user
**Builds on:** `2026-07-07-voice-companion-design.md` (v1, merged to master)

## Problem

Two complaints from the first real use of v1:

1. The Piper voice (`en_US-lessac-medium`) sounds robotic.
2. The companion's personality is cold — the system prompt only says
   "friendly and encouraging", which llama3.1 renders as bland politeness.

## Decisions

### 1. Replace Piper with Kokoro TTS

- New dependency `kokoro-onnx`; `piper-tts` is removed. (Upstream examples
  also install `soundfile`, but that is only for saving WAV files — we play
  through `sounddevice` directly and don't need it.)
- Kokoro runs on CPU through onnxruntime. This is deliberate: it leaves the
  12 GB of VRAM for Whisper and llama3.1, and Kokoro is faster than
  real-time on CPU.
- Voice: `am_michael` (calm male), chosen by the user. Configurable.
- API (verified against upstream `examples/save.py`, 2026-07-08):

  ```python
  from kokoro_onnx import Kokoro
  kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
  samples, sample_rate = kokoro.create(text, voice="am_michael", speed=1.0, lang="en-us")
  ```

  `samples` is a float32 numpy array; play with `sounddevice` directly —
  no WAV round-trip needed (simpler than the Piper path).
- Model files (one-time download, ~330 MB total), placed at repo root:
  - `https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx`
  - `https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin`

### 2. Speaker interface unchanged

`Speaker` keeps its public shape — construct with paths from config, then
`.speak(text)` blocks until playback finishes. `main.py` changes only in
which config constants it passes. `speak_safely` still wraps failures.

### 3. Config changes (`companion/config.py`)

- Remove `PIPER_VOICE_PATH`.
- Add `KOKORO_MODEL_PATH = "kokoro-v1.0.onnx"`,
  `KOKORO_VOICES_PATH = "voices-v1.0.bin"`,
  `KOKORO_VOICE = "am_michael"`, `KOKORO_SPEED = 1.0`.
- Rewrite `SYSTEM_PROMPT`: a real character — warm, playful, genuinely
  curious about the user's life; reacts to what the user says instead of
  just answering; light humor; encouragement that feels earned. Keeps the
  v1 requirements: open question at session start, adapt to the chosen
  focus, short spoken-style replies.
- Rewrite `GREETING` to match the personality.

### 4. Startup check

`check_voice_file_available()` becomes a check that **both** Kokoro files
exist, with a README pointer in the error message.

### 5. Docs and hygiene

- README step 4 rewritten for the two Kokoro downloads; requirements
  pinned (`kokoro-onnx` with a compatible range).
- `.gitignore`: add `voices-v1.0.bin` (`*.onnx` already covered).

## Testing

- `tests/test_speaker.py` rewritten: mock `Kokoro`, assert `create()` is
  called with configured voice/speed and that playback uses the returned
  samples and sample rate.
- All other tests (21 passing today) must keep passing.
- Final acceptance is manual: user listens and judges voice + personality.

## Out of scope

Streaming/low-latency TTS, barge-in, voice cloning, and the v2 backlog
items from the v1 final review.
