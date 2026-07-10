# Crash-Safe Raw Log — Design Spec

**Date:** 2026-07-10
**Status:** Approved by user
**Builds on:** Structured memory (`2026-07-08-structured-memory-design.md`), which
noted as a known limitation: "memory is written only on 'bye bye'. Closing the
terminal mid-session loses that session." This spec closes that gap.

## Problem

`remember_session()` (in `companion/session.py`) is the only thing that ever
writes conversation content to disk, and it only runs on a graceful exit — the
sleep phrase, or the stop-flag flush at the end of `run_loop`. The raw
conversation itself (`LLMClient.turns`) lives in memory only.

If the process dies abruptly — a crashed process, a killed terminal, or (the
triggering incident) a power loss — `turns` is gone. The conversation is lost
with no way to recover it, even though the user actually had it.

## Decisions

### 1. `RawLog`: an incremental, crash-safe transcript

New class in `companion/memory.py`, alongside `Memory`. One JSONL file per
wake→sleep session, written turn-by-turn as the conversation happens, so a
crash at any point only loses turns that hadn't been spoken yet.

- **Location:** `memory/raw/`, sibling to `durable.md`/`timeline.md`, created
  on first write.
- **Filename:** `%Y%m%d-%H%M%S.jsonl` (e.g. `20260710-230115.jsonl`) — no
  colons (Windows-safe), lexicographically sortable = chronological.
- **Format:** one JSON object per line, `{"role": "user"|"assistant",
  "content": "..."}` — the same shape as `LLMClient.turns`, so a recovered
  file can be dropped straight into a fresh `LLMClient`.
- **Durability:** each `append()` does `write()` + `flush()` +
  `os.fsync()`, so a completed append survives a power loss a moment later.

```python
class RawLog:
    def __init__(self, dir_path: str): ...
    def start(self) -> None: ...          # create this session's file
    def append(self, role: str, content: str) -> None: ...
    def finish(self) -> None: ...         # delete this session's file

    @staticmethod
    def pending(dir_path: str) -> list[str]: ...        # orphan paths, oldest first
    @staticmethod
    def load(path: str) -> list[dict]: ...              # parsed turns; skips a
                                                          # trailing corrupt line
    @staticmethod
    def delete(path: str) -> None: ...
```

`RawLog` is pure file I/O — like `Memory`, it has no LLM knowledge and no
opinion about *when* it's called.

### 2. `remember_session()` reports success

Currently returns nothing and just emits warnings on failure. It changes to
return `bool`: `True` if the session had nothing to save, or everything it
tried to save succeeded; `False` if either the timeline append or the durable
merge raised. Callers use this to decide whether it's safe to delete the raw
log — a failed save must not destroy the only remaining copy of the
conversation.

```python
def remember_session(llm, memory, emit) -> bool:
    if not llm.has_user_turns():
        return True
    ...
    ok = True
    try:
        memory.append_timeline(...)
    except Exception as exc:
        emit(...); ok = False
    try:
        memory.write_durable(...)
    except Exception as exc:
        emit(...); ok = False
    return ok
```

### 3. Write path — wired into `run_loop`

- **`Action.WAKE`:** create a new `RawLog`, `start()` it, `append("assistant",
  GREETING)`.
- **`Action.FORWARD`, after a successful `llm.send()`:** `append("user",
  text)` then `append("assistant", reply)` — mirrors exactly what gets added
  to `llm.turns`, so recovered turns reconstruct the real conversation.
- **`Action.SLEEP`, and the end-of-run flush when stopped mid-conversation:**
  call `remember_session()` as today; call `raw_log.finish()` only if it
  returned `True`.

### 4. Recovery path — wired into `run_session()`

Runs once at startup, after `Memory` is constructed, before the "Ready"
message:

```python
for path in RawLog.pending(config.RAW_LOG_DIR):
    turns = RawLog.load(path)
    recovery_llm = LLMClient(llm.provider, config.SYSTEM_PROMPT)
    recovery_llm.turns = turns
    if remember_session(recovery_llm, memory, emit):
        RawLog.delete(path)
```

Processed oldest-first, so multiple crashes before a clean restart all get
folded in, in order. Reuses the live session's already-constructed
`llm.provider` rather than building a new one. A system event is emitted only
if at least one orphan was found and processed — silent on the common case.

### 5. Config (`companion/config.py`)

```python
RAW_LOG_DIR = os.path.join(MEMORY_DIR, "raw")
```

## Error handling & edge cases

- **Save fails (recovery or normal goodbye):** raw file is kept; retried on
  the next startup. No data lost, just delayed.
- **Truncated/corrupt last line** (the realistic failure mode for an actual
  power cut mid-write): `RawLog.load()` parses line-by-line and skips a line
  that fails to parse rather than discarding the whole file.
- **Orphan with only the greeting, no user turns:** `remember_session()`
  already no-ops (`has_user_turns()` false) and returns `True` — treated as
  success, file deleted, nothing to summarize.
- **Multiple orphan files:** all recovered in one startup pass, oldest first.

## Testing

- `RawLog` unit tests (new `tests/test_memory.py` cases): `start()` creates
  the file; `append()` writes valid JSONL; `finish()` deletes it;
  `pending()`/`load()` find and parse orphan files, skipping a corrupt
  trailing line; `delete()` removes a specific file.
- `remember_session()` unit tests: `True` on success; `False` if either
  write raises; `True` (no-op) when there are no user turns.
- `test_session.py` integration tests: a full wake→message→sleep cycle
  writes then deletes the raw file; a run stopped mid-conversation leaves the
  raw file if the flush's `remember_session()` fails, deletes it if it
  succeeds.
- Recovery test: pre-seed `memory/raw/` with an orphan file, run
  `run_session`, confirm `durable.md`/`timeline.md` are updated, the orphan
  file is gone, and a recovery system event was emitted.
- Existing test suite keeps passing.

## Out of scope

- Recovering a session that crashed *during* recovery itself (already
  covered: the file simply stays pending for the next startup).
- Exposing raw transcripts in the UI/timeline (they're a recovery mechanism,
  not a user-facing feature).
- Any change to the *structured* memory model (`durable.md`/`timeline.md`
  format is unchanged).
