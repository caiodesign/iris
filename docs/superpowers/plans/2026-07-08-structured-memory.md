# Structured Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat `memory.md` log with structured memory — a durable, LLM-merged knowledge base (Facts/Goals/English) that is always loaded in full, plus an append-only dated timeline loaded as a recent tail — so the companion reliably recalls durable facts and recent history.

**Architecture:** A `memory/` folder holds two files: `durable.md` (three markdown sections the LLM rewrites/merges at session end) and `timeline.md` (dated entries appended each session). The `Memory` class exposes `load()` (assembled prompt block), `load_durable()` (raw durable for the merge input), `append_timeline()`, and `write_durable()` (empty-guarded, atomic). At session end, `main.py` makes two side-channel `llm.summarize()` calls — one for the timeline entry, one for the durable merge — via a new testable `remember_session(llm, memory)` helper.

**Tech Stack:** Python 3, stdlib only (`os`, `datetime`), pytest + `unittest.mock`.

## Global Constraints

- **No new dependencies** — stdlib `os` and `datetime` only (matches the existing `memory.py`).
- **`memory/` is git-ignored user data** — `.gitignore` changes from `memory.md` to `memory/`.
- **No migration** — no pre-existing `memory.md` on disk; empty state must behave exactly like today (no memory section added to the prompt).
- **`load()` returns the body only** — `llm_client.reset()` already wraps memory with the header "What you remember about the user from previous sessions:", so `load()` must NOT repeat that header.
- **Durable overwrite is guarded** — `write_durable` ignores empty/whitespace input and writes atomically (temp file + `os.replace`).
- **TDD** — write the failing test first, watch it fail, implement minimally, watch it pass.
- **Commit messages** end with the trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

### Task 1: Rewrite the `Memory` class for structured memory

**Files:**
- Modify (rewrite): `companion/memory.py`
- Modify (rewrite): `tests/test_memory.py`

**Interfaces:**
- Consumes: nothing (stdlib only; constructor takes literal args, no `config` import).
- Produces (later tasks rely on these exact signatures):
  - `Memory(dir_path: str, timeline_max_chars: int)`
  - `Memory.load() -> str` — assembled durable + recent-timeline block, `""` when both empty. No "What you remember" header.
  - `Memory.load_durable() -> str` — raw `durable.md` contents, stripped; `""` if missing.
  - `Memory.append_timeline(entry: str) -> None` — appends `## <timestamp>\n<entry>\n\n` to `timeline.md`.
  - `Memory.write_durable(text: str) -> None` — overwrites `durable.md` atomically; no-op if `text` is empty/whitespace.

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/test_memory.py` with:

```python
# tests/test_memory.py
import os

from companion.memory import Memory


def _memory(tmp_path, timeline_max_chars=4000):
    return Memory(str(tmp_path / "memory"), timeline_max_chars)


def test_load_returns_empty_string_when_nothing_stored(tmp_path):
    assert _memory(tmp_path).load() == ""


def test_load_durable_returns_empty_string_when_file_missing(tmp_path):
    assert _memory(tmp_path).load_durable() == ""


def test_append_timeline_writes_dated_heading_and_entry(tmp_path):
    memory = _memory(tmp_path)

    memory.append_timeline("- Talked about food.")

    content = (tmp_path / "memory" / "timeline.md").read_text(encoding="utf-8")
    assert content.startswith("## 20")  # "## 2026-07-08 14:30" style heading
    assert "- Talked about food." in content


def test_append_timeline_accumulates_sessions(tmp_path):
    memory = _memory(tmp_path)

    memory.append_timeline("- First session.")
    memory.append_timeline("- Second session.")

    content = (tmp_path / "memory" / "timeline.md").read_text(encoding="utf-8")
    assert "- First session." in content
    assert "- Second session." in content
    assert content.count("## 20") == 2


def test_write_durable_persists_and_load_durable_reads_it_back(tmp_path):
    memory = _memory(tmp_path)

    memory.write_durable("## Facts\n- Caio likes ramen.")

    assert memory.load_durable() == "## Facts\n- Caio likes ramen."


def test_write_durable_overwrites_previous_content(tmp_path):
    memory = _memory(tmp_path)

    memory.write_durable("## Facts\n- Old fact.")
    memory.write_durable("## Facts\n- New fact.")

    assert memory.load_durable() == "## Facts\n- New fact."


def test_write_durable_ignores_empty_or_whitespace_input(tmp_path):
    memory = _memory(tmp_path)

    memory.write_durable("## Facts\n- Keep me.")
    memory.write_durable("   \n  ")  # must NOT blank out accumulated facts

    assert memory.load_durable() == "## Facts\n- Keep me."


def test_write_durable_leaves_no_temp_file_behind(tmp_path):
    memory = _memory(tmp_path)

    memory.write_durable("## Facts\n- Caio likes ramen.")

    files = os.listdir(tmp_path / "memory")
    assert files == ["durable.md"]  # atomic rename left no .tmp artifact


def test_load_returns_durable_only_when_no_timeline(tmp_path):
    memory = _memory(tmp_path)

    memory.write_durable("## Facts\n- Caio likes ramen.")

    assert memory.load() == "## Facts\n- Caio likes ramen."


def test_load_assembles_durable_then_recent_sessions(tmp_path):
    memory = _memory(tmp_path)

    memory.write_durable("## Facts\n- Caio likes ramen.")
    memory.append_timeline("- Talked about food.")

    loaded = memory.load()
    assert loaded == (
        "## Facts\n- Caio likes ramen.\n\n"
        "Recent sessions:\n"
        + (tmp_path / "memory" / "timeline.md").read_text(encoding="utf-8").strip()
    )
    # Durable comes first, timeline second under a "Recent sessions:" header.
    assert loaded.index("## Facts") < loaded.index("Recent sessions:")


def test_load_truncates_timeline_to_the_tail_keeping_the_end(tmp_path):
    memory = Memory(str(tmp_path / "memory"), timeline_max_chars=20)

    memory.append_timeline("B" * 40)
    loaded = memory.load()

    # Only the timeline is windowed; it keeps the END of the file.
    assert "Recent sessions:\n" in loaded
    tail = loaded.split("Recent sessions:\n", 1)[1]
    assert len(tail) <= 20
    assert tail == "B" * 20
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_memory.py -v`
Expected: FAIL — `TypeError` / `AttributeError` (old `Memory` has `append_session`/`load(path,max_chars)` shape, not the new methods).

- [ ] **Step 3: Rewrite the implementation**

Replace the entire contents of `companion/memory.py` with:

```python
# companion/memory.py
import os
from datetime import datetime


class Memory:
    """Structured memory: a durable, LLM-managed knowledge base plus a dated,
    append-only timeline.

    - durable.md holds Facts / Goals / English sections the LLM rewrites and
      merges at session end. It is loaded in full so it never scrolls out.
    - timeline.md holds one dated entry per session, appended. Only the recent
      tail (timeline_max_chars) is loaded.
    """

    def __init__(self, dir_path: str, timeline_max_chars: int):
        self.dir_path = dir_path
        self.timeline_max_chars = timeline_max_chars
        self.durable_path = os.path.join(dir_path, "durable.md")
        self.timeline_path = os.path.join(dir_path, "timeline.md")

    def load(self) -> str:
        # Body only: llm_client.reset() supplies the "What you remember..."
        # header, so repeating it here would double up.
        durable = self.load_durable()
        timeline = self._load_timeline_tail()
        parts = []
        if durable:
            parts.append(durable)
        if timeline:
            parts.append("Recent sessions:\n" + timeline)
        return "\n\n".join(parts)

    def load_durable(self) -> str:
        if not os.path.exists(self.durable_path):
            return ""
        with open(self.durable_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def append_timeline(self, entry: str) -> None:
        os.makedirs(self.dir_path, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(self.timeline_path, "a", encoding="utf-8") as f:
            f.write(f"## {stamp}\n{entry.strip()}\n\n")

    def write_durable(self, text: str) -> None:
        # Empty-guard: a blank or failed merge must never wipe accumulated
        # facts. Atomic temp+rename: a crash mid-write can't truncate the file.
        if not text.strip():
            return
        os.makedirs(self.dir_path, exist_ok=True)
        tmp_path = self.durable_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")
        os.replace(tmp_path, self.durable_path)

    def _load_timeline_tail(self) -> str:
        if not os.path.exists(self.timeline_path):
            return ""
        with open(self.timeline_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return text[-self.timeline_max_chars :]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_memory.py -v`
Expected: PASS (all 11 tests).

- [ ] **Step 5: Run the full suite to confirm no regressions elsewhere**

Run: `pytest -q`
Expected: `test_main.py` still references the old memory wiring only via `choose_from_menu` (unaffected); everything except any `main.py` memory usage passes. `main.py` still imports fine. Note any failures — they should be limited to nothing here (main.py is untouched in this task).

- [ ] **Step 6: Commit**

```bash
git add companion/memory.py tests/test_memory.py
git commit -m "feat: structured Memory class (durable + timeline)

Rewrite Memory around two files: durable.md (Facts/Goals/English,
loaded in full, overwritten with an empty-guard + atomic rename) and
timeline.md (dated append, loaded as a recent tail). load() returns the
assembled body without the 'What you remember' header, which reset()
still supplies.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Wire config, `.gitignore`, and `main.py` session-end flow

**Files:**
- Modify: `companion/config.py` (memory constants + prompts)
- Modify: `.gitignore` (`memory.md` → `memory/`)
- Modify: `companion/main.py` (Memory construction + `remember_session` helper + SLEEP handler)
- Modify: `tests/test_main.py` (add `remember_session` tests)

**Interfaces:**
- Consumes (from Task 1): `Memory(dir_path, timeline_max_chars)`, `Memory.load()`, `Memory.load_durable()`, `Memory.append_timeline(entry)`, `Memory.write_durable(text)`.
- Produces:
  - `config.MEMORY_DIR: str`, `config.TIMELINE_MAX_CHARS: int`, `config.TIMELINE_PROMPT: str`, `config.DURABLE_MERGE_PROMPT: str`
  - `main.remember_session(llm, memory) -> None` — session-end persistence: appends a timeline entry and merges durable memory, each guarded independently.

- [ ] **Step 1: Update `companion/config.py`**

In `companion/config.py`, replace the memory block. Find:

```python
MEMORY_PATH = "memory.md"
MEMORY_MAX_CHARS = 6000

SUMMARY_PROMPT = (
    "The session is over. Summarize it in 3 to 5 short bullet points for "
    "your own memory before the next session: topics discussed, English "
    "mistakes the user made, and personal facts you learned about the user "
    "(trips, food, family, work, plans). Write only the bullet points."
)
```

Replace with:

```python
MEMORY_DIR = "memory"
# Recent-tail window for the dated timeline. Durable memory (Facts/Goals/
# English) is always loaded in full, so it has no char cap here.
TIMELINE_MAX_CHARS = 4000

TIMELINE_PROMPT = (
    "The session is over. Summarize it in 2 to 4 short bullet points for "
    "your timeline: what you talked about and any notable moments. Write "
    "only the bullet points."
)

# Merge instruction for durable memory. The current durable.md contents are
# appended after this text before the call.
DURABLE_MERGE_PROMPT = (
    "The session is over. Below is what you currently remember about the "
    "user, in three sections: Facts (durable personal facts), Goals (what he "
    "wants to work on or talk about), and English (recurring mistakes and "
    "focus areas). Update this memory using the session: keep every fact "
    "that is still true, add anything new you learned, refine or remove only "
    "what this session directly contradicts, and merge duplicates. Keep it "
    "concise. Return the full updated memory as exactly those three markdown "
    "sections (## Facts, ## Goals, ## English) and nothing else; keep a "
    "section's heading even if it has no bullets yet.\n\n"
    "Current memory:\n"
)
```

- [ ] **Step 2: Update `.gitignore`**

In `.gitignore`, change the line `memory.md` to:

```
memory/
```

- [ ] **Step 3: Write the failing tests for `remember_session`**

Append to `tests/test_main.py`. First update the import line at the top:

```python
from unittest.mock import MagicMock, patch

from companion.main import (
    PROVIDER_NAMES,
    STT_NAMES,
    choose_from_menu,
    remember_session,
)
```

Then add these tests at the end of the file:

```python
def test_remember_session_appends_timeline_and_merges_durable():
    llm = MagicMock()
    llm.has_user_turns.return_value = True
    # First summarize call -> timeline entry, second -> merged durable memory.
    llm.summarize.side_effect = ["- Talked about food.", "## Facts\n- Likes ramen."]
    memory = MagicMock()
    memory.load_durable.return_value = "## Facts\n- Old fact."

    remember_session(llm, memory)

    assert llm.summarize.call_count == 2
    memory.append_timeline.assert_called_once_with("- Talked about food.")
    memory.write_durable.assert_called_once_with("## Facts\n- Likes ramen.")


def test_remember_session_does_nothing_without_user_turns():
    llm = MagicMock()
    llm.has_user_turns.return_value = False
    memory = MagicMock()

    remember_session(llm, memory)

    llm.summarize.assert_not_called()
    memory.append_timeline.assert_not_called()
    memory.write_durable.assert_not_called()


def test_remember_session_still_merges_durable_when_timeline_call_fails():
    llm = MagicMock()
    llm.has_user_turns.return_value = True
    llm.summarize.side_effect = [Exception("network blip"), "## Facts\n- Likes ramen."]
    memory = MagicMock()
    memory.load_durable.return_value = ""

    remember_session(llm, memory)  # must not raise

    memory.append_timeline.assert_not_called()
    memory.write_durable.assert_called_once_with("## Facts\n- Likes ramen.")


def test_remember_session_keeps_timeline_when_durable_call_fails():
    llm = MagicMock()
    llm.has_user_turns.return_value = True
    llm.summarize.side_effect = ["- Talked about food.", Exception("network blip")]
    memory = MagicMock()
    memory.load_durable.return_value = ""

    remember_session(llm, memory)  # must not raise

    memory.append_timeline.assert_called_once_with("- Talked about food.")
    memory.write_durable.assert_not_called()
```

- [ ] **Step 4: Run the new tests to verify they fail**

Run: `pytest tests/test_main.py -v`
Expected: FAIL — `ImportError: cannot import name 'remember_session'`.

- [ ] **Step 5: Add the `remember_session` helper to `companion/main.py`**

In `companion/main.py`, add this helper next to the other module-level helpers (e.g. directly after `speak_safely`):

```python
def remember_session(llm, memory) -> None:
    if not llm.has_user_turns():
        return
    print("Remembering this session...")
    # Two independent side-channel calls: a failure in one must not skip the
    # other, and neither may crash the goodbye (mirrors the send/tts guards).
    try:
        memory.append_timeline(llm.summarize(config.TIMELINE_PROMPT))
    except Exception as exc:
        print(f"WARNING: Could not update timeline memory ({exc}).")
    try:
        merged = llm.summarize(config.DURABLE_MERGE_PROMPT + memory.load_durable())
        memory.write_durable(merged)
    except Exception as exc:
        print(f"WARNING: Could not update durable memory ({exc}).")
```

- [ ] **Step 6: Update the `Memory` construction in `main()`**

In `companion/main.py`, find:

```python
    memory = Memory(config.MEMORY_PATH, config.MEMORY_MAX_CHARS)
```

Replace with:

```python
    memory = Memory(config.MEMORY_DIR, config.TIMELINE_MAX_CHARS)
```

- [ ] **Step 7: Replace the SLEEP handler's persistence block**

In `companion/main.py`, find the SLEEP branch:

```python
            elif action == Action.SLEEP:
                print("Going back to sleep.")
                speak_safely(speaker, "Bye for now!")
                if llm.has_user_turns():
                    print("Remembering this session...")
                    try:
                        memory.append_session(llm.summarize(config.SUMMARY_PROMPT))
                    except Exception as exc:
                        print(f"WARNING: Could not save session memory ({exc}).")
```

Replace with:

```python
            elif action == Action.SLEEP:
                print("Going back to sleep.")
                speak_safely(speaker, "Bye for now!")
                remember_session(llm, memory)
```

- [ ] **Step 8: Run the new tests to verify they pass**

Run: `pytest tests/test_main.py -v`
Expected: PASS (existing `choose_from_menu` tests + 4 new `remember_session` tests).

- [ ] **Step 9: Run the full suite**

Run: `pytest -q`
Expected: PASS — all tests green, output pristine (no warnings). Confirm the count went up by the new tests and nothing regressed.

- [ ] **Step 10: Commit**

```bash
git add companion/config.py companion/main.py tests/test_main.py .gitignore
git commit -m "feat: session-end structured-memory wiring

Add MEMORY_DIR/TIMELINE_MAX_CHARS and the timeline + durable-merge
prompts; point Memory at the memory/ folder; git-ignore memory/.
Replace the single append-summary at sleep with remember_session(),
which appends a timeline entry and merges durable memory in two
independently-guarded side-channel calls.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Update README documentation

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the behavior built in Tasks 1–2.
- Produces: nothing (docs only).

- [ ] **Step 1: Locate the memory documentation in `README.md`**

Run: `grep -n -i "memory\|remember\|memory.md" README.md`
Expected: find the sentence(s) describing how the companion remembers sessions (the old flat-`memory.md` behavior), likely under "Usage notes" or similar.

- [ ] **Step 2: Update the memory description**

Replace the existing memory description with wording that reflects the new model. Use this text (adapt surrounding heading/format to match the file's existing style):

```markdown
### Memory

The companion keeps two kinds of memory in a git-ignored `memory/` folder:

- `durable.md` — a knowledge base of **Facts**, **Goals**, and **English**
  focus areas. It is loaded in full at the start of every session and is
  rewritten and merged by the brain when you say goodbye, so durable facts
  never age out.
- `timeline.md` — one dated entry per session. Only the most recent portion
  (`TIMELINE_MAX_CHARS`) is loaded, so old sessions naturally fade.

Both files are plain markdown you can read or hand-edit. Delete the `memory/`
folder to start fresh.
```

- [ ] **Step 3: Verify no stale references remain**

Run: `grep -n -i "memory.md\|MEMORY_PATH\|MEMORY_MAX_CHARS\|SUMMARY_PROMPT" README.md`
Expected: no matches (all stale references removed).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document structured memory (durable + timeline)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Empty first run:** with no `memory/` folder, `load()` returns `""` and the wake handler's `llm.reset("")` adds no memory section — identical to today. The first durable merge runs with an empty `Current memory:` and the brain creates the three sections from scratch.
- **Why `load()` omits the header:** `companion/llm_client.py`'s `reset()` prepends "What you remember about the user from previous sessions:\n" to whatever `load()` returns. Do not add a competing header inside `load()`.
- **Do not** import `config` inside `memory.py` — the `Memory` constructor takes its paths/limits as arguments (keeps it unit-testable with `tmp_path`, matching the existing test style).
