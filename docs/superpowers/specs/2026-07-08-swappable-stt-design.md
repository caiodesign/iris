# Swappable STT — Local or Cloud Transcription (v1.4) — Design

**Date:** 2026-07-08
**Status:** Approved by user
**Builds on:** v1.3 (cloud brains), v1.2.1 (thinking room), v1.2 (memory), v1.1 (Kokoro TTS)

## Problem

Transcription ("ears") is hard-wired to local faster-whisper. The user — a
non-native English speaker practicing conversation — wants the option of a
cloud transcription API (OpenAI's `gpt-4o-transcribe` family), which has
lower word-error rates on accents and noisy rooms than classic Whisper,
while keeping local whisper as the free, private, gaming-friendly default.

This is the direct parallel of v1.3's cloud brains: the same launch-time,
user-chosen switch, applied to the STT step instead of the thinking step.
TTS (Kokoro) stays local. The brain choice and the ears choice are
**independent** — any combination is valid (e.g. local llama brain + cloud
ears, or cloud brain + local ears).

## Decisions

### 1. Ears selection (mirrors the brain selection)

- `STT_PROVIDER = "local"` in config — the Enter-default. Valid values:
  `"local"`, `"openai"`.
- **Interactive menu at startup** (user requirement): a second numbered
  menu after the brain menu — `local` (default marked) / `openai` — reading
  one line; empty input = config default. Enter-Enter at launch reproduces
  today's all-local behavior.
- **`--ears <name>` CLI flag** skips the ears menu
  (`python -m companion.main --brain claude --ears openai`).
  Flag > menu > config, same precedence as `--brain`.
- Selecting `openai` ears is independent of the brain: it does not load or
  require Ollama, and does not change the brain.

### 2. Shared menu helper (refactor of existing brain menu)

`choose_brain()` in `main.py` currently hard-codes its menu loop. Extract
the shared logic into one generic helper and call it for both selections:

```python
def choose_from_menu(label, names, default, cli_choice) -> str: ...

brain = choose_from_menu("brain", PROVIDER_NAMES, config.LLM_PROVIDER, args.brain)
ears  = choose_from_menu("ears",  STT_NAMES,      config.STT_PROVIDER, args.ears)
```

This is a behavior-preserving refactor: number picks, name picks,
Enter→default, and unknown→default all keep their current semantics. The
existing brain-menu behavior (and its tests) must remain green.

### 3. Model default (user-chosen, config-editable)

| Ears | Config constant | Default | Cost |
|---|---|---|---|
| local  | `WHISPER_MODEL_SIZE` / `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` | `base.en` on `cuda`/`float16` | free |
| openai | `OPENAI_TRANSCRIBE_MODEL` | `gpt-4o-transcribe` (full) | ≈$0.006/min |

Full transcribe is the default (best accuracy on accents/noise, which
matters most for a learner). Switching the constant to
`gpt-4o-mini-transcribe` halves the cost. At 30 min/day, full transcribe is
≈$5/mo; mini ≈$2.70/mo. Existing `WHISPER_*` constants are untouched and
continue to drive the local backend.

### 4. `companion/transcriber.py` becomes an adapter module

The module mirrors `providers.py`: one duck-typed interface, two
implementations, one factory. The interface is unchanged from today, so the
main loop's call site is behavior-compatible:

```python
class <X>Transcriber:
    def transcribe(self, audio: np.ndarray) -> str: ...

def make_transcriber(name: str) -> object   # "local" | "openai"
```

- **LocalTranscriber** — today's `Transcriber`, renamed verbatim. Keeps the
  faster-whisper model, the `beam_size=5` decode, and the Windows
  `_register_pip_nvidia_dll_dirs()` cuBLAS/cuDNN fix. No behavior change.
- **OpenAITranscriber(model)** — constructs `OpenAI()` (reads
  `OPENAI_API_KEY` from env, same as the OpenAI brain). `transcribe`
  encodes the audio to an in-memory WAV and calls
  `client.audio.transcriptions.create(model=..., file=("speech.wav",
  wav_bytes, "audio/wav"))`, returning `result.text.strip()`.
- `make_transcriber` is the only place the class names are wired; unknown
  name raises `ValueError` (parallels `make_provider`).

### 5. Audio encoding (standard library, no new dependency)

The captured audio is `np.float32` mono at 16 kHz; the API wants a file.
A helper `_encode_wav(audio) -> bytes` uses the stdlib `wave` + `io.BytesIO`:
clip samples to [-1, 1], scale to 16-bit PCM, write a WAV container in
memory. The `openai` package is already a dependency (v1.3 brain), so this
whole feature adds **no** new packages.

### 6. API key check and model loading

- If `ears == "openai"`, verify `OPENAI_API_KEY` is set at startup and exit
  with a friendly, variable-naming error if missing — reusing the same key
  check that guards a cloud brain, so a missing key fails fast before the
  first utterance rather than mid-session. (`OPENAI_API_KEY` may be needed
  by the brain, the ears, or both; a single presence check per variable.)
- `load_transcriber(ears)` branches: `local` keeps today's behavior,
  including the zero-array **warm-up** that surfaces CUDA/cuDNN problems
  early with the README hint; `openai` skips the warm-up (no GPU model to
  load) and just constructs the client.

### 7. Mid-session resilience for transcription

Today `main.py` calls `transcriber.transcribe(audio)` with no guard — safe
for local whisper, but a cloud call can fail on a network blip and would
crash the session. Wrap it like the existing `llm.send` guard: on any
exception, print a `WARNING` and `continue` (drop that utterance, keep
listening). This also hardens the local path against a bad frame.

Behavior note: cloud ears add a network round-trip (~0.5–2s) per utterance
before the reply. The "cancel that" mechanic is unchanged — cancel is
evaluated after the text returns, so ordering is preserved; the window
just shifts slightly later.

### 8. Docs and hygiene

- `config.py`: add `STT_PROVIDER = "local"` and
  `OPENAI_TRANSCRIBE_MODEL = "gpt-4o-transcribe"`.
- README: new "Cloud ears (optional)" note beside "Cloud brains (optional)"
  — the `--ears` flag and startup menu, the cost ballpark (≈$5/mo full,
  ≈$2.70/mo mini), and that it reuses the existing `OPENAI_API_KEY`.
- No `requirements.txt` change (openai already present).

## Testing

TDD, mirroring `test_transcriber.py` and `test_providers.py`:

- `make_transcriber("local")` / `("openai")` return the right classes;
  unknown name raises `ValueError`.
- `_encode_wav` produces a valid WAV (RIFF header, 16 kHz, mono, 16-bit) and
  round-trips a known array back to matching int16 samples.
- `OpenAITranscriber.transcribe` with a **mocked** OpenAI client: asserts it
  passes a WAV file tuple to `audio.transcriptions.create` with the
  configured model and returns `.text` stripped. No real API calls.
- `choose_from_menu`: number picks, name picks, Enter→default, junk→default
  — a behavior-preserving refactor; existing brain-menu tests stay green.
- Loop resilience: a transcriber that raises is swallowed and the loop
  continues.
- Smoke test: one real exchange with `--ears openai` if a key is present.

## Out of scope

Streaming transcription, language hints / prompt-priming params, per-ears
retries or backoff beyond the SDK defaults, a second cloud STT provider,
mid-session ears switching by voice, and usage/cost tracking. The adapter
shape leaves room to add another cloud STT later without touching the loop.
