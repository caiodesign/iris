# Memory + Warmth (v1.2) — Design

**Date:** 2026-07-08
**Status:** Approved by user
**Builds on:** v1 design (`2026-07-07-voice-companion-design.md`, which sketched
this memory feature as "documented now, built later") and v1.1
(`2026-07-08-voice-personality-upgrade-design.md`).

## Problem

1. The companion has no memory: every "hey chat" starts from scratch, so it
   re-asks about things the user already told it (their trip, food).
2. The personality is better after v1.1 but still reads a bit cold. The user
   chose the "warm, curious friend" direction (Claude-like: caring, genuinely
   interested, remembers details, playful but not over the top).

## Decisions

### 1. Session-summary memory (`memory.md`)

The model summarizes; we never persist raw transcripts (user's explicit
requirement).

- **On sleep ("bye bye"):** after speaking the farewell, make one extra LLM
  call asking for a 3–5 bullet summary of the session: topics discussed,
  English mistakes the user made, personal facts learned (trips, food,
  family, work…). Append it to `memory.md` under a dated heading.
  - Skip the summary entirely if the user never said anything after the
    greeting (no user turns in history).
  - Any failure (Ollama down, disk error) is caught, printed as a WARNING,
    and never crashes the app — worst case one session is forgotten.
- **On wake ("hey chat"):** read `memory.md` and pass it into the LLM's
  system prompt as remembered context ("What you remember about the user
  from previous sessions: …"). Missing/empty file → no memory section.
- **Pruning:** inject only the most recent `MEMORY_MAX_CHARS` (6000) characters
  (whole file is kept on disk; truncation happens at load, cutting from the
  front at a section boundary where practical — simple char slice is
  acceptable).
- `memory.md` lives at the repo root, is human-readable/editable, and is
  git-ignored (personal data, not code).

**Rejected alternatives:** full transcript persistence (noisy, grows fast,
user explicitly doesn't want it); vector store/embeddings (overkill for one
user, one machine).

**Known limitation (accepted):** memory is written only on "bye bye".
Closing the terminal mid-session loses that session.

### 2. New module: `companion/memory.py`

```python
class Memory:
    def __init__(self, path: str, max_chars: int): ...
    def load(self) -> str:
        """Tail of memory.md up to max_chars; '' if file missing/empty."""
    def append_session(self, summary: str) -> None:
        """Append '## <YYYY-MM-DD HH:MM>' heading + summary + blank line."""
```

No LLM knowledge in this module — it is pure file I/O.

### 3. `LLMClient` extensions (`companion/llm_client.py`)

- `reset(memory: str = "")` — re-seeds history with the system prompt; if
  `memory` is non-empty, the system message becomes
  `SYSTEM_PROMPT + "\n\nWhat you remember about the user from previous sessions:\n" + memory`.
- `summarize() -> str` — one-off `ollama.chat` call: current history plus an
  appended user instruction asking for the 3–5 bullet summary. The
  instruction and reply are NOT stored in `self.history` (the session is
  over). Returns the reply text.

### 4. Config (`companion/config.py`)

- `USER_NAME = "Caio"`
- `MEMORY_PATH = "memory.md"`
- `MEMORY_MAX_CHARS = 6000`
- `SUMMARY_PROMPT` — the summarization instruction (single source of truth).
- `SYSTEM_PROMPT` rewritten for "warm, curious friend": knows the user's
  name (interpolates `USER_NAME`) and uses it naturally; explicitly told to
  bring up remembered details unprompted ("how was that trip?"); keeps all
  v1.1 constraints (open question at session start, adapt to chosen focus,
  1–3 spoken-style sentences, no lists/markdown/emojis/stage directions).

### 5. `main.py` wiring

- Construct `Memory(config.MEMORY_PATH, config.MEMORY_MAX_CHARS)` at startup.
- WAKE: `llm.reset(memory.load())` then seed/speak greeting as today.
- SLEEP: speak "Bye for now!" first (snappy exit), then summarize + append,
  each wrapped so failure only WARNs. Print "Remembering this session..."
  so the extra pause is explained.

## Testing

- `tests/test_memory.py`: append creates file with dated heading; append
  twice accumulates; load returns '' when missing; load truncates to
  max_chars keeping the end.
- `tests/test_llm_client.py`: reset with memory injects the combined system
  prompt; reset without memory unchanged; summarize sends history + summary
  instruction without mutating history and returns reply text.
- Existing 21 tests keep passing.
- Acceptance: user has a session, says "bye bye", checks `memory.md`
  content, starts a new session and sees the companion reference it.

## Out of scope

Mid-session memory writes (crash safety), LLM-generated dynamic greetings,
memory compaction/rewriting, everything else on the v2 list.
