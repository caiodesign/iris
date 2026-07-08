# Switchable Brains — Cloud LLM Providers (v1.3) — Design

**Date:** 2026-07-08
**Status:** Approved by user
**Builds on:** v1.2.1 (thinking room), v1.2 (memory), v1.1 (Kokoro TTS)

## Problem

The local llama3.1:8b brain (a) competes with games for the 12 GB of VRAM
and (b) lacks the knowledge/quality for topics like Elden Ring build advice.
The user wants to optionally route the conversation to a cloud LLM — Claude,
OpenAI, or z.ai (GLM) — while keeping local llama as the default, free brain.
STT (Whisper) and TTS (Kokoro) always stay local; only the "thinking" step
switches.

## Decisions

### 1. Provider selection

- `LLM_PROVIDER = "local"` in config — the default used when the user just
  presses Enter. Valid values: `"local"`, `"claude"`, `"openai"`, `"zai"`.
- **Interactive menu at startup** (user requirement): right after launch,
  print a numbered menu (local / claude / openai / zai, with the config
  default marked) and read one line; empty input = config default.
- **`--brain <name>` CLI flag** skips the menu entirely
  (`python -m companion.main --brain claude`). Flag > menu > config.
- In any cloud mode the app never touches Ollama (no startup check, no
  model load) — VRAM stays free for games. Whisper (~1 GB) still uses GPU.

### 2. API keys via `.env` (user requirement)

- Git-ignored `.env` file at the repo root holds
  `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `ZAI_API_KEY`.
- Loaded at startup with `python-dotenv` (`load_dotenv()`), which puts them
  in the process environment where the SDKs look for them.
- Startup check in cloud mode: if the selected provider's env var is
  missing, exit with a friendly error naming the exact variable and the
  README section. No key material ever appears in code or git.

### 3. Models (user-chosen defaults, config-editable)

| Provider | Config constant | Default | Price in/out per MTok |
|---|---|---|---|
| local  | `OLLAMA_MODEL` | `llama3.1:8b` | free |
| claude | `ANTHROPIC_MODEL` | `claude-sonnet-5` | $3/$15 (intro $2/$10) |
| openai | `OPENAI_MODEL` | `gpt-5.4` | $2.50/$15 |
| zai    | `ZAI_MODEL` | `glm-5` | $0.60/$1.92 |

`CLOUD_MAX_TOKENS = 1024` (replies are 1–3 spoken sentences; summaries are
3–5 bullets). Ballpark cost: roughly half a cent to 2 cents per exchange
on Sonnet 5 / gpt-5.4, well under half a cent on GLM-5. The README states
this plainly. Exact z.ai base URL and model ID string are verified against
docs.z.ai during implementation.

### 4. New module: `companion/providers.py`

One small adapter per provider behind a single duck-typed interface:

```python
class <X>Provider:
    def chat(self, system: str, turns: list[dict]) -> str: ...

def make_provider(name: str) -> object   # "local"|"claude"|"openai"|"zai"
REQUIRED_ENV = {"claude": "ANTHROPIC_API_KEY", ...}   # local absent
```

- **LocalProvider** — `ollama.chat(model, messages=[system]+turns)`
  (current behavior, moved).
- **ClaudeProvider** — official `anthropic` SDK,
  `client.messages.create(model, max_tokens, system=..., messages=turns)`.
  No sampling params, no thinking config. Reply = concatenated `text`
  blocks. Claude requires the first message to be `user`, but sessions
  start with the seeded assistant greeting — the adapter folds any leading
  assistant turns into the system text
  (`'You already opened this session by saying: "..."'`) for each request.
  `stop_reason == "refusal"` raises a `RuntimeError` (caught by main's
  guard, see §6).
- **OpenAICompatProvider(model, api_key_env, base_url=None)** — official
  `openai` SDK; serves both OpenAI (no base_url) and z.ai (their
  OpenAI-compatible endpoint). Reply =
  `response.choices[0].message.content`.

### 5. `LLMClient` refactor (public interface unchanged)

Constructor becomes `LLMClient(provider, system_prompt)`. Internally the
system prompt and the turn list are stored separately (`self.system`,
`self.turns`) instead of one combined history — required by Claude's API
shape and harmless for the others. `send`, `reset(memory)`,
`summarize(instruction)`, `has_user_turns`, `seed_assistant` keep their
exact signatures and semantics; stage-direction stripping stays here
(provider-agnostic).

### 6. Mid-session resilience (also closes a v2 backlog item)

Cloud APIs hiccup (network, rate limits) and Ollama can die mid-session.
`main.py` wraps the `llm.send(...)` call: on any exception it prints a
WARNING with the error and speaks
`"Sorry, I had trouble thinking. Say that again?"` — the session continues
instead of crashing.

### 7. Docs and hygiene

- `.gitignore`: add `.env`.
- `requirements.txt`: add `anthropic`, `openai`, `python-dotenv` (pinned
  ranges).
- README: new "Cloud brains (optional)" section — where to get each key,
  `.env` format, the startup menu and `--brain` flag, the cost ballpark,
  and the gaming note (cloud mode leaves VRAM free).

## Testing

- `tests/test_providers.py`: ClaudeProvider (system passed through, leading
  assistant greeting folded into system, text blocks joined, refusal
  raises); OpenAICompatProvider (base_url and key wiring, message shape,
  reply extraction); LocalProvider (system+turns concatenation) — all with
  mocked SDKs.
- `tests/test_llm_client.py`: rewritten around an injected fake provider;
  same behaviors covered as today (history, reset+memory, summarize
  non-mutation, has_user_turns, stage-direction stripping).
- Existing state-machine/memory/speaker/transcriber tests unaffected.
- Smoke test: one real exchange per provider the user has keys for.

## Out of scope

Mid-session brain switching by voice, streaming responses, per-provider
personalities, usage/cost tracking, retries/backoff beyond the SDKs'
defaults.
