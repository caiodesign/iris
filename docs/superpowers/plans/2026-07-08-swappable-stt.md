# Swappable STT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user pick local faster-whisper or OpenAI cloud transcription at launch, exactly the way the brain is already picked.

**Architecture:** Turn `companion/transcriber.py` into an adapter module mirroring `companion/providers.py` — one `transcribe(audio) -> str` interface, a `LocalTranscriber` and an `OpenAITranscriber`, and a `make_transcriber(name)` factory. `main.py` gains a generic `choose_from_menu` helper (replacing the brain-specific `choose_brain`), a second "ears" menu, and a `--ears` flag. TTS and the brain are untouched; the brain and ears choices are independent.

**Tech Stack:** Python, faster-whisper (local STT), the `openai` SDK (cloud STT, already a dependency for the brain), stdlib `wave`/`io` for in-memory WAV encoding, pytest + `unittest.mock`.

## Global Constraints

- **No new dependencies.** `openai` is already in `requirements.txt`; cloud WAV encoding uses only stdlib `wave` + `io`. Do not add packages.
- **Local behavior must stay identical** when ears = local: same `WhisperModel(...)` construction, `beam_size=5` decode, Windows NVIDIA-DLL registration, and startup warm-up.
- **Default cloud model is `gpt-4o-transcribe`** (verified live 2026-07-08). Config-editable to `gpt-4o-mini-transcribe`.
- **Unit tests never hit real APIs, GPUs, or the network** — mock `WhisperModel` and `OpenAI` at the `companion.transcriber` module path, as `test_providers.py` mocks its SDKs.
- **Every commit message ends with the trailer** `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Run the full suite with `pytest -q` from the repo root.

---

### Task 1: Rename `Transcriber` → `LocalTranscriber` and add the factory

**Files:**
- Modify: `companion/transcriber.py`
- Modify: `companion/main.py:82-102` (`load_transcriber`) and the import at `companion/main.py:17`
- Test: `tests/test_transcriber.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `LocalTranscriber(model_size, device, compute_type)` with `.transcribe(audio: np.ndarray) -> str`; `make_transcriber(name: str) -> object` handling `"local"` and raising `ValueError` on anything else.

- [ ] **Step 1: Update the existing transcriber tests to the new class name**

In `tests/test_transcriber.py`, change the import and the three constructor calls from `Transcriber` to `LocalTranscriber`:

```python
from companion.transcriber import LocalTranscriber
```
Then replace each `Transcriber(` with `LocalTranscriber(` in `test_transcribe_joins_and_strips_segment_text`, `test_transcribe_returns_empty_string_for_silence`, and `test_init_registers_pip_nvidia_dll_dirs_before_loading_model`.

- [ ] **Step 2: Add the factory tests**

Append to `tests/test_transcriber.py`:

```python
import pytest

from companion.transcriber import make_transcriber


def test_make_transcriber_builds_local():
    with patch("companion.transcriber.WhisperModel"):
        transcriber = make_transcriber("local")
    assert isinstance(transcriber, LocalTranscriber)


def test_make_transcriber_rejects_unknown_name():
    with pytest.raises(ValueError):
        make_transcriber("robot-ears")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_transcriber.py -q`
Expected: FAIL — `ImportError: cannot import name 'LocalTranscriber'` / `make_transcriber`.

- [ ] **Step 4: Rename the class and add the factory**

In `companion/transcriber.py`, rename `class Transcriber:` to `class LocalTranscriber:` (body unchanged) and add the factory at the end of the file:

```python
def make_transcriber(name: str):
    from companion import config

    if name == "local":
        return LocalTranscriber(
            config.WHISPER_MODEL_SIZE,
            config.WHISPER_DEVICE,
            config.WHISPER_COMPUTE_TYPE,
        )
    raise ValueError(f"Unknown transcriber: {name}")
```

- [ ] **Step 5: Update `main.py` to the new class name (keep behavior identical)**

In `companion/main.py`, change the import on line 17 from `from companion.transcriber import Transcriber` to `from companion.transcriber import LocalTranscriber`. In `load_transcriber`, change the constructor call and the return annotation:

```python
def load_transcriber() -> LocalTranscriber:
    try:
        transcriber = LocalTranscriber(
            config.WHISPER_MODEL_SIZE, config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE
        )
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
```

- [ ] **Step 6: Run the full suite to verify it passes**

Run: `pytest -q`
Expected: PASS (all tests green).

- [ ] **Step 7: Commit**

```bash
git add companion/transcriber.py companion/main.py tests/test_transcriber.py
git commit -m "refactor: rename Transcriber to LocalTranscriber, add make_transcriber factory"
```

---

### Task 2: In-memory WAV encoder

**Files:**
- Modify: `companion/transcriber.py`
- Test: `tests/test_transcriber.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_encode_wav(audio: np.ndarray, sample_rate: int = 16000) -> bytes` — a 16 kHz mono 16-bit PCM WAV container.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_transcriber.py`:

```python
import io
import wave

from companion.transcriber import _encode_wav


def test_encode_wav_is_16k_mono_pcm16_and_round_trips():
    samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)

    data = _encode_wav(samples, sample_rate=16000)

    assert data[:4] == b"RIFF"
    with wave.open(io.BytesIO(data), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        frames = wav.readframes(wav.getnframes())

    decoded = np.frombuffer(frames, dtype="<i2")
    expected = np.array([0, 16383, -16383, 32767, -32767], dtype="<i2")
    assert np.array_equal(decoded, expected)


def test_encode_wav_clips_out_of_range_samples():
    data = _encode_wav(np.array([2.0, -2.0], dtype=np.float32))
    with wave.open(io.BytesIO(data), "rb") as wav:
        decoded = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert np.array_equal(decoded, np.array([32767, -32767], dtype="<i2"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_transcriber.py -k encode_wav -q`
Expected: FAIL — `ImportError: cannot import name '_encode_wav'`.

- [ ] **Step 3: Implement the encoder**

At the top of `companion/transcriber.py`, add `import io` and `import wave` to the existing imports. Then add the function above the class definitions:

```python
def _encode_wav(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    # The OpenAI transcription endpoint wants an audio file, but we hold a
    # float32 mono array in [-1, 1]. Clip, scale to 16-bit PCM, and wrap it in
    # a WAV container in memory — stdlib only, so no extra dependency.
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
    return buffer.getvalue()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_transcriber.py -k encode_wav -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add companion/transcriber.py tests/test_transcriber.py
git commit -m "feat: add in-memory WAV encoder for cloud transcription"
```

---

### Task 3: `OpenAITranscriber` + wire `"openai"` into the factory

**Files:**
- Modify: `companion/transcriber.py`
- Modify: `companion/config.py:12-15` (add the transcription-model constant near the other model names)
- Test: `tests/test_transcriber.py`

**Interfaces:**
- Consumes: `_encode_wav` (Task 2); `config.OPENAI_TRANSCRIBE_MODEL` (added here).
- Produces: `OpenAITranscriber(model: str)` with `.transcribe(audio) -> str`; `make_transcriber("openai")` returning it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transcriber.py`:

```python
from companion.transcriber import OpenAITranscriber


def test_openai_transcriber_sends_wav_and_strips_reply():
    fake_result = MagicMock()
    fake_result.text = "  Hello there  "
    with patch("companion.transcriber.OpenAI") as MockOpenAI:
        client = MockOpenAI.return_value
        client.audio.transcriptions.create.return_value = fake_result

        transcriber = OpenAITranscriber("gpt-4o-transcribe")
        result = transcriber.transcribe(np.zeros(16000, dtype=np.float32))

    assert result == "Hello there"
    kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-transcribe"
    filename, data, mime = kwargs["file"]
    assert filename == "speech.wav"
    assert mime == "audio/wav"
    assert data[:4] == b"RIFF"


def test_make_transcriber_builds_openai():
    with patch("companion.transcriber.OpenAI"):
        transcriber = make_transcriber("openai")
    assert isinstance(transcriber, OpenAITranscriber)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_transcriber.py -k openai -q`
Expected: FAIL — `ImportError: cannot import name 'OpenAITranscriber'`.

- [ ] **Step 3: Add the config constant**

In `companion/config.py`, directly below `OPENAI_MODEL = "gpt-5.4"` (line 12), add:

```python
OPENAI_TRANSCRIBE_MODEL = "gpt-4o-transcribe"  # cloud "ears"; "gpt-4o-mini-transcribe" halves cost
```

- [ ] **Step 4: Implement the transcriber and extend the factory**

In `companion/transcriber.py`, add `from openai import OpenAI` to the imports. Add the class after `LocalTranscriber`:

```python
class OpenAITranscriber:
    def __init__(self, model: str):
        self.client = OpenAI()  # reads OPENAI_API_KEY from env, like the brain
        self.model = model

    def transcribe(self, audio: np.ndarray) -> str:
        wav_bytes = _encode_wav(audio)
        result = self.client.audio.transcriptions.create(
            model=self.model,
            file=("speech.wav", wav_bytes, "audio/wav"),
        )
        return result.text.strip()
```

Then add this branch inside `make_transcriber`, before the final `raise`:

```python
    if name == "openai":
        return OpenAITranscriber(config.OPENAI_TRANSCRIBE_MODEL)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_transcriber.py -q`
Expected: PASS (all transcriber tests).

- [ ] **Step 6: Commit**

```bash
git add companion/transcriber.py companion/config.py tests/test_transcriber.py
git commit -m "feat: add OpenAITranscriber cloud STT backend"
```

---

### Task 4: Generic launch menu + `--ears` flag

**Files:**
- Modify: `companion/main.py` (`PROVIDER_NAMES` area lines 20; `choose_brain` lines 41-56; argparse lines 114-122)
- Modify: `companion/config.py` (add `STT_PROVIDER`)
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `choose_from_menu(label, names, default, cli_choice) -> str`; module constant `STT_NAMES = ["local", "openai"]`; `--ears` CLI flag. `choose_brain` is removed.

- [ ] **Step 1: Rewrite the menu tests around the generic helper**

Replace the entire contents of `tests/test_main.py` with:

```python
# tests/test_main.py
from unittest.mock import patch

from companion.main import PROVIDER_NAMES, STT_NAMES, choose_from_menu


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_main.py -q`
Expected: FAIL — `ImportError: cannot import name 'choose_from_menu'` / `STT_NAMES`.

- [ ] **Step 3: Add the config default**

In `companion/config.py`, directly below `LLM_PROVIDER = "local"` (line 9), add:

```python
# Which transcription backend ("ears"): "local" (faster-whisper) or "openai".
STT_PROVIDER = "local"
```

- [ ] **Step 4: Replace `choose_brain` with the generic helper and add `STT_NAMES`**

In `companion/main.py`, below `PROVIDER_NAMES = ["local", "claude", "openai", "zai"]` (line 20), add:

```python
STT_NAMES = ["local", "openai"]
```

Delete the whole `choose_brain` function (lines 41-56) and replace it with:

```python
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
```

- [ ] **Step 5: Add the `--ears` flag (leave the call sites for Task 5)**

In `main()`, add the `--ears` argument next to `--brain`:

```python
    parser.add_argument(
        "--ears",
        choices=STT_NAMES,
        default=None,
        help="skip the ears menu and use this transcription backend",
    )
```

Then update the brain selection line to call the new helper (the ears call site is added in Task 5):

```python
    args = parser.parse_args()
    brain = choose_from_menu("brain", PROVIDER_NAMES, config.LLM_PROVIDER, args.brain)
    print(f"Brain: {brain}")
```

- [ ] **Step 6: Run the full suite to verify it passes**

Run: `pytest -q`
Expected: PASS. Also confirm the flag registered:

Run: `python -m companion.main --help`
Expected: help text lists both `--brain` and `--ears`.

- [ ] **Step 7: Commit**

```bash
git add companion/main.py companion/config.py tests/test_main.py
git commit -m "refactor: generic launch menu, add --ears flag and STT_PROVIDER"
```

---

### Task 5: Wire ears into startup (menu, key check, model load, loop guard)

**Files:**
- Modify: `companion/main.py` (`load_transcriber` lines 82-102; import line 17; `main()` body)

**Interfaces:**
- Consumes: `make_transcriber` (Task 1/3), `choose_from_menu` + `--ears` (Task 4), `config.STT_PROVIDER` (Task 4), `check_api_key_available` (existing).
- Produces: `load_transcriber(ears: str)`; the running app now transcribes with the chosen backend and survives a transcription failure.

This task changes `main()`'s imperative startup flow, which has no unit harness today (the existing `llm.send` guard is likewise unit-test-free). It is verified by the full suite staying green plus the smoke checks in Step 6.

- [ ] **Step 1: Switch the transcriber import to the factory**

In `companion/main.py` line 17, change `from companion.transcriber import LocalTranscriber` to:

```python
from companion.transcriber import make_transcriber
```

- [ ] **Step 2: Make `load_transcriber` take the ears choice**

Replace the whole `load_transcriber` function with:

```python
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
```

- [ ] **Step 3: Choose ears, check its key, and load it**

In `main()`, right after the brain selection/print, add the ears selection:

```python
    ears = choose_from_menu(
        "ears (transcription)", STT_NAMES, config.STT_PROVIDER, args.ears
    )
    print(f"Ears: {ears}")
```

In the "Checking services and microphone..." block, after the brain key/Ollama check and before `check_microphone_available()`, add:

```python
    if ears == "openai":
        # OpenAI cloud ears reuse the brain's OPENAI_API_KEY; fail fast if
        # absent (REQUIRED_ENV["openai"] == "OPENAI_API_KEY").
        check_api_key_available("openai")
```

Then change the transcriber construction from `transcriber = load_transcriber()` to:

```python
    transcriber = load_transcriber(ears)
```

- [ ] **Step 4: Guard transcription in the main loop**

In the `while True` loop, replace:

```python
            audio = detector.listen_for_utterance()
            text = transcriber.transcribe(audio)
            if not text:
                continue
```

with:

```python
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
```

- [ ] **Step 5: Run the full suite to verify nothing regressed**

Run: `pytest -q`
Expected: PASS (all tests green).

- [ ] **Step 6: Smoke-check the wiring**

Run: `python -m companion.main --brain local --ears local`
Expected: startup prints `Brain: local` then `Ears: local`, loads whisper, reaches `Ready. Say "hey chat" to start.` Ctrl-C to exit.

If an `OPENAI_API_KEY` is present in `.env`, also run: `python -m companion.main --brain local --ears openai`
Expected: startup prints `Ears: openai`, does **not** warm up a GPU model, reaches `Ready`. Say a sentence; confirm `Heard: ...` appears (transcribed via the API).

To confirm the missing-key path: temporarily unset the key and run `--ears openai`; expected a clean `ERROR: OPENAI_API_KEY is not set...` and exit, not a traceback.

- [ ] **Step 7: Commit**

```bash
git add companion/main.py
git commit -m "feat: select and load transcription backend at launch, guard transcription"
```

---

### Task 6: Document cloud ears

**Files:**
- Modify: `README.md` (add a subsection after the "Cloud brains (optional)" block, before "Usage notes")

**Interfaces:** none (docs only).

- [ ] **Step 1: Add the "Cloud ears (optional)" section**

In `README.md`, immediately after the "Cloud brains (optional)" section and before "## Usage notes", insert:

```markdown
## Cloud ears (optional)

By default the companion transcribes your speech locally with
faster-whisper (free, private, works offline). You can instead route
transcription to OpenAI, which recognizes accents and noisy rooms more
accurately than local Whisper — at a small per-minute cost.

1. Reuse the same `OPENAI_API_KEY` from "Cloud brains" (no extra key).
2. Pick the ears at launch — the app shows a second menu after the brain
   menu, or skip it with:
   ```
   python -m companion.main --ears openai
   ```
   The brain and ears are independent: `--brain local --ears openai` runs
   the free local llama brain with cloud transcription.
3. The model is set in `companion/config.py` (`OPENAI_TRANSCRIBE_MODEL`,
   default `gpt-4o-transcribe`). Change it to `gpt-4o-mini-transcribe` to
   roughly halve the cost. Ballpark at 30 min/day: about $5/month on full
   transcribe, about $2.70/month on mini.
```

- [ ] **Step 2: Verify the section reads correctly**

Run: `grep -n "Cloud ears" README.md`
Expected: one match, positioned after the "Cloud brains" section.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document optional cloud ears (OpenAI transcription)"
```

---

## Self-Review

**Spec coverage** (against `2026-07-08-swappable-stt-design.md`):
- §1 Ears selection (menu, `--ears`, precedence, independence) → Task 4 + Task 5.
- §2 Shared `choose_from_menu` refactor (behavior-preserving) → Task 4.
- §3 Model default `gpt-4o-transcribe`, config-editable → Task 3 (config) + Task 6 (mini note).
- §4 Adapter module (`LocalTranscriber`, `OpenAITranscriber`, `make_transcriber`) → Tasks 1 + 3.
- §5 Stdlib WAV encoding, no new dependency → Task 2 (+ Global Constraints).
- §6 Key check + `load_transcriber(ears)` warm-up branch → Task 5.
- §7 Loop resilience + latency note → Task 5 (guard) + Task 6 (behavior context).
- §8 Docs + config constants, no `requirements.txt` change → Tasks 3, 4, 6.

**Placeholder scan:** none — every code step shows complete code; every command shows expected output.

**Type consistency:** `make_transcriber(name)`, `LocalTranscriber(model_size, device, compute_type)`, `OpenAITranscriber(model)`, `_encode_wav(audio, sample_rate=16000) -> bytes`, `choose_from_menu(label, names, default, cli_choice)`, `load_transcriber(ears)` are used identically wherever they appear across tasks. `STT_NAMES`, `config.STT_PROVIDER`, and `config.OPENAI_TRANSCRIBE_MODEL` are defined before their consumers.
