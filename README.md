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
4. Download the Kokoro voice model (two files, ~330 MB total, run from the
   project root so they land here):
   ```
   curl -L -o kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
   curl -L -o voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
   ```
   To change the voice or speaking speed, edit `KOKORO_VOICE` / `KOKORO_SPEED`
   in `companion/config.py` (voice list: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).
5. Run it:
   ```
   python -m companion.main
   ```

## Cloud brains (optional)

By default the companion thinks with the free local llama model. You can
also route the conversation to a cloud model — better answers (game builds,
niche topics) and no VRAM use, but it costs real money per exchange
(ballpark: half a cent to 2 cents per exchange on Sonnet 5 / gpt-5.4,
far less on GLM-5).

1. Create a file named `.env` in the project root (it is git-ignored):
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   OPENAI_API_KEY=sk-...
   ZAI_API_KEY=...
   ```
   Only add the keys you have: Claude → console.anthropic.com,
   OpenAI → platform.openai.com, z.ai → z.ai (API keys page).
2. Pick the brain at launch — the app shows a menu, or skip it with:
   ```
   python -m companion.main --brain claude
   ```
3. Models are set in `companion/config.py` (`ANTHROPIC_MODEL`,
   `OPENAI_MODEL`, `ZAI_MODEL`). Cloud mode never loads the llama model,
   so your GPU stays free for games (only Whisper uses ~1 GB).

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

## Usage notes

- Say "Cancel That" **in the same breath** as the sentence you want to
  retract (e.g., "I went to... cancel that"). Once you pause, the app has
  already sent what you said to the model and a reply is on its way.
- The companion keeps two kinds of memory in a git-ignored `memory/` folder:
  - `durable.md` — a knowledge base of **Facts**, **Goals**, and **English**
    focus areas. It is loaded in full at the start of every session and is
    rewritten and merged by the brain when you say goodbye, so durable facts
    never age out.
  - `timeline.md` — one dated entry per session. Only the most recent portion
    (`TIMELINE_MAX_CHARS`) is loaded, so old sessions naturally fade.
  Both files are plain markdown you can read or hand-edit. Delete the
  `memory/` folder to start fresh.
