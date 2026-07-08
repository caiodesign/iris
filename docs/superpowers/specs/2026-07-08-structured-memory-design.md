# Structured Memory — Design Spec

**Date:** 2026-07-08
**Status:** Approved
**Feature:** B (structured memory) of the three-feature roadmap. Follows Feature A
(swappable STT). Feature C (Conversation Coach) is a separate later cycle.

## Problem

Memory today is a single flat markdown file (`memory.md`). At session end the LLM
emits 3–5 bullets that are appended under a `## <timestamp>` heading; at wake the
last `MEMORY_MAX_CHARS` (6000) characters are injected into the system prompt.

Two consequences:

1. **Durable facts silently scroll out.** Truncation keeps only the end of the
   file, so durable knowledge (family, work, goals, recurring English mistakes)
   ages out of the window over time even though it never stops being true.
2. **Everything is one undifferentiated log.** There is no structure the companion
   can lean on to *proactively* recall — bring up a past trip, resume an English
   focus area — which is the "Her"-like behavior we're aiming for.

**North star:** proactive recall. The companion should reliably know durable facts
and recent history so it can bring things up on its own. Structure is the mechanism
that makes that possible.

## Core idea: two kinds of memory

Memory is split by lifetime, and each kind is loaded and updated differently:

- **Durable & bounded** — always loaded in full, so it never scrolls out:
  **Facts**, **Goals** (with "ideas / wants to talk about" folded in), **English**
  (recurring mistakes and focus areas).
- **Chronological & windowed** — only the recent tail is loaded: **Timeline**
  (dated session entries). This is where "how did that trip go?" comes from — fresh,
  then it ages out.

## Decisions

### 1. File layout

A git-ignored `memory/` folder with two files:

**`memory/durable.md`** — the always-loaded, LLM-managed knowledge base, three
markdown sections the model owns and rewrites:

```
## Facts
- Caio is a software engineer at Nubank.
- Visited Japan in spring 2026; loved the food.

## Goals
- Wants to speak more fluently in meetings.
- Would like to talk about sci-fi movies sometime.

## English
- Recurring: drops articles ("I went to store").
- Working on past-tense irregular verbs.
```

**`memory/timeline.md`** — append-only, dated entries:

```
## 2026-07-08 14:30
- Talked about his weekend trip; practiced past tense.
```

"Ideas" is folded into Goals rather than given its own section (YAGNI — it is a
small, closely related list). Facts/Goals/English stay as sections within one
`durable.md` (not separate files) because the merge is one holistic call over all
three; splitting them would only add reassembly work.

`.gitignore` changes from `memory.md` to `memory/`.

### 2. Update flow (session end / "bye bye")

Two side-channel LLM calls, both via the existing `llm.summarize(instruction)`
mechanism. This fires once per session (when the user says goodbye), never per turn.

**Call 1 — Timeline entry (append).** Instruction (`TIMELINE_PROMPT`): summarize
this session in 2–4 bullets for the timeline (topics, notable moments). Result goes
to `memory.append_timeline(entry)`, written under a fresh `## <timestamp>` heading.
Pure append, unchanged in spirit from today.

**Call 2 — Durable merge (rewrite).** Instruction (`DURABLE_MERGE_PROMPT`) includes
the **current `durable.md` contents** plus guidance: this is what you currently
remember; update it with anything new or changed from this session; keep every
still-true fact; refine or remove only what this session contradicts; dedupe; stay
concise; return the full updated Facts / Goals / English sections. Result goes to
`memory.write_durable(new_text)`, which overwrites `durable.md`.

**Why two calls, not one combined call:** each returns plain text with nothing to
delimiter-parse, so each is independently robust and trivially testable. Session-end
latency is not user-facing (the companion already said goodbye), and this runs once
per session, so the extra call is cheap even on a cloud brain.

### 3. Load flow (wake / "hey chat")

`memory.load()` assembles the two files into one block for the system prompt:

```
What you remember about the user:

<full contents of durable.md>

Recent sessions:

<tail of timeline.md, up to TIMELINE_MAX_CHARS, most recent kept>
```

- **Durable loaded in full** — bounded by the merge prompt keeping it concise.
- **Timeline tail only** — same "keep the end" truncation the current code uses,
  scoped to timeline.

`main.py`'s wake handler is unchanged in shape: `llm.reset(memory.load())` then
`seed_assistant(GREETING)`. The system prompt already instructs the model to bring
things up itself, so proactive recall follows from reliably having durable facts +
recent timeline present.

**Empty state:** first run, both files missing → `load()` returns `""`, and
`reset("")` behaves exactly as today (no memory section added). No migration:
`memory.md` is git-ignored and does not currently exist on disk.

### 4. Resilience & safety

The merge (Call 2) is the only operation that overwrites existing knowledge, so it
carries the guardrails. Everything else is append-only or read-only and cannot lose
data.

- **Empty-guard on `write_durable`.** If the merge returns empty or whitespace-only
  text, skip the write and keep the existing `durable.md`. A blank/failed response
  must never blank out accumulated facts.
- **Per-call exception isolation.** The two session-end calls are wrapped
  independently (mirroring today's `try/except` around `append_session`): timeline
  failure still attempts the durable merge; durable-merge failure leaves `durable.md`
  untouched; neither crashes the session or blocks the goodbye. Warnings:
  `WARNING: Could not update timeline memory (...)` and
  `WARNING: Could not update durable memory (...)`.
- **Atomic write.** `write_durable` writes to a temp file and renames over
  `durable.md`, so a crash mid-write cannot leave a half-truncated file.
  `timeline.md` is append-only and already safe.

Deliberately **not** included (YAGNI for a single-user local tool): versioned
backups, rollback, per-fact confidence scores, git safety net.

### 5. Config & API

**`config.py`:**

```python
MEMORY_DIR = "memory"              # replaces MEMORY_PATH = "memory.md"
TIMELINE_MAX_CHARS = 4000          # recent-tail window for timeline load
# MEMORY_MAX_CHARS removed (durable loads in full; timeline uses the above)

TIMELINE_PROMPT = "..."            # Call 1 instruction (session -> dated entry)
DURABLE_MERGE_PROMPT = "..."       # Call 2 instruction prefix; durable.md appended to it
# SUMMARY_PROMPT removed (replaced by the two above)
```

**`Memory` class (new API):**

```python
Memory(dir_path, timeline_max_chars)
  load() -> str            # assembled durable + recent-timeline block for the prompt
  load_durable() -> str    # raw durable.md, fed into the merge instruction
  append_timeline(entry)   # dated "## <ts>" append to timeline.md
  write_durable(text)      # empty-guarded, atomic temp+rename overwrite of durable.md
```

Creates `memory/` on first write. `main.py`'s session-end block calls `summarize`
twice and routes the results to `append_timeline` / `write_durable`, each in its own
`try/except`.

### 6. Testing

pytest + `tmp_path`, no real LLM — same style as the existing `test_memory.py`:

- `load()` empty state → `""`; durable-only; durable + timeline assembled in order.
- `load()` truncates timeline to the tail (keeps the end), leaves durable full.
- `append_timeline` writes a dated heading; accumulates across sessions.
- `write_durable` overwrites; **empty/whitespace input is ignored** (old content
  survives) — the key safety test.
- `write_durable` is atomic (no partial file; temp cleaned up).
- `main.py` session-end wiring: both `summarize` calls fire, results routed
  correctly, one call failing does not block the other (mocked `llm`).

## Out of scope

- Conversation Coach (Feature C) — separate spec → plan → build cycle.
- Mid-session memory edits / commands.
- Semantic / embedding-based retrieval.
- Per-fact timestamps or confidence scoring.
- Migrating any pre-existing flat `memory.md` (none exists on disk).
