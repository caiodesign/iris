# Crash-Safe Raw Log Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Iris never silently loses a conversation to a crash or power loss — every turn is written to disk as it happens, and any conversation orphaned by an abrupt exit gets folded into memory automatically the next time Iris starts.

**Architecture:** A new `RawLog` class (`companion/memory.py`) writes one fsync'd JSONL file per wake-to-sleep session under `memory/raw/`. `companion/session.py`'s `run_loop` writes to it turn-by-turn and deletes it after a successful `remember_session()`. A new `recover_orphaned_sessions()` runs once at `run_session()` startup, replaying any leftover files through the same `remember_session()` path before the interactive loop begins.

**Tech Stack:** Python 3.13, pytest (mock-based unit tests, `tmp_path` fixture, no real filesystem/LLM dependencies in tests).

## Global Constraints

- Raw log files live at `memory/raw/` (already covered by the existing `memory/` gitignore entry — no `.gitignore` change needed).
- Filenames are `%Y%m%d-%H%M%S.jsonl` — no colons (Windows-safe), lexicographically sortable = chronological.
- Each JSONL line is exactly `{"role": "user"|"assistant", "content": "..."}` — the same shape as `LLMClient.turns`.
- `remember_session()` changes from returning `None` to returning `bool` (`True` = safe to delete the raw log; `False` = keep it for retry). This is a breaking signature change for its two existing call sites in `run_loop`, both updated in this plan.
- `run_loop()`'s parameter order becomes `(capture, transcriber, llm, memory, speaker, machine, raw_log, emit, should_stop)` — `raw_log` inserted right before `emit`.
- Full spec: `docs/superpowers/specs/2026-07-10-crash-safe-raw-log-design.md`.

---

### Task 1: `RawLog` class

**Files:**
- Modify: `companion/memory.py` (add `import json`, add `RawLog` class after `Memory`)
- Test: `tests/test_memory.py` (add `import json`, add `RawLog` tests after the existing `Memory` tests)

**Interfaces:**
- Produces:
  - `RawLog(dir_path: str)` — instance, reused across a whole `run_session()` call.
  - `raw_log.start() -> None` — creates a new empty session file at `raw_log.path`.
  - `raw_log.append(role: str, content: str) -> None` — appends one JSONL line, flushed + fsynced.
  - `raw_log.finish() -> None` — deletes `raw_log.path`, resets it to `None`.
  - `RawLog.pending(dir_path: str) -> list[str]` — sorted (oldest-first) list of orphan file paths; `[]` if the directory doesn't exist.
  - `RawLog.load(path: str) -> list[dict]` — parsed turns; a corrupt/truncated trailing line is skipped, not fatal.
  - `RawLog.delete(path: str) -> None` — removes a specific file.

- [ ] **Step 1: Write the failing tests**

Add to the top of `tests/test_memory.py`:

```python
import json
```

(keep the existing `import os` line; the module already has one)

Change the import line:

```python
from companion.memory import Memory, RawLog
```

Append to the end of `tests/test_memory.py`:

```python
def _raw_log(tmp_path):
    return RawLog(str(tmp_path / "memory" / "raw"))


def test_start_creates_an_empty_file(tmp_path):
    raw_log = _raw_log(tmp_path)

    raw_log.start()

    assert os.path.exists(raw_log.path)
    with open(raw_log.path, encoding="utf-8") as f:
        assert f.read() == ""


def test_append_writes_one_json_line_per_turn(tmp_path):
    raw_log = _raw_log(tmp_path)
    raw_log.start()

    raw_log.append("assistant", "Hey there!")
    raw_log.append("user", "Hi!")

    with open(raw_log.path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert json.loads(lines[0]) == {"role": "assistant", "content": "Hey there!"}
    assert json.loads(lines[1]) == {"role": "user", "content": "Hi!"}


def test_finish_deletes_the_session_file(tmp_path):
    raw_log = _raw_log(tmp_path)
    raw_log.start()
    raw_log.append("user", "hi")
    path = raw_log.path

    raw_log.finish()

    assert not os.path.exists(path)
    assert raw_log.path is None


def test_pending_returns_empty_list_when_dir_missing(tmp_path):
    assert RawLog.pending(str(tmp_path / "memory" / "raw")) == []


def test_pending_lists_orphan_files_oldest_first(tmp_path):
    dir_path = tmp_path / "memory" / "raw"
    dir_path.mkdir(parents=True)
    (dir_path / "20260710-090000.jsonl").write_text(
        '{"role": "user", "content": "a"}\n', encoding="utf-8"
    )
    (dir_path / "20260709-090000.jsonl").write_text(
        '{"role": "user", "content": "b"}\n', encoding="utf-8"
    )

    pending = RawLog.pending(str(dir_path))

    assert [os.path.basename(p) for p in pending] == [
        "20260709-090000.jsonl",
        "20260710-090000.jsonl",
    ]


def test_load_parses_turns_in_order(tmp_path):
    dir_path = tmp_path / "memory" / "raw"
    dir_path.mkdir(parents=True)
    path = dir_path / "session.jsonl"
    path.write_text(
        '{"role": "assistant", "content": "Hey!"}\n'
        '{"role": "user", "content": "Hi!"}\n',
        encoding="utf-8",
    )

    assert RawLog.load(str(path)) == [
        {"role": "assistant", "content": "Hey!"},
        {"role": "user", "content": "Hi!"},
    ]


def test_load_skips_a_corrupt_trailing_line(tmp_path):
    dir_path = tmp_path / "memory" / "raw"
    dir_path.mkdir(parents=True)
    path = dir_path / "session.jsonl"
    path.write_text(
        '{"role": "user", "content": "good line"}\n'
        '{"role": "user", "conte',  # truncated, as a real power cut would leave it
        encoding="utf-8",
    )

    assert RawLog.load(str(path)) == [{"role": "user", "content": "good line"}]


def test_delete_removes_the_file(tmp_path):
    dir_path = tmp_path / "memory" / "raw"
    dir_path.mkdir(parents=True)
    path = dir_path / "session.jsonl"
    path.write_text("{}\n", encoding="utf-8")

    RawLog.delete(str(path))

    assert not os.path.exists(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_memory.py -v`
Expected: `ImportError: cannot import name 'RawLog'` (or collection error) since `RawLog` doesn't exist yet.

- [ ] **Step 3: Implement `RawLog`**

In `companion/memory.py`, add `import json` to the top imports (alongside the existing `import os` and `from datetime import datetime`), then append the class after `Memory`:

```python
class RawLog:
    """Incremental, crash-safe transcript for one wake-to-sleep session.

    Written turn-by-turn during the conversation (unlike Memory, which is
    only ever updated at a graceful goodbye) so a crash mid-session loses at
    most the turn that hadn't finished writing yet. Deleted once its
    contents have been folded into Memory; a leftover file at startup means
    the previous run never got there.
    """

    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        self.path = None

    def start(self) -> None:
        os.makedirs(self.dir_path, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(self.dir_path, f"{stamp}.jsonl")
        open(self.path, "w", encoding="utf-8").close()

    def append(self, role: str, content: str) -> None:
        # flush + fsync so a completed append survives a power loss a
        # moment later, not just a process crash.
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"role": role, "content": content}) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def finish(self) -> None:
        if self.path and os.path.exists(self.path):
            os.remove(self.path)
        self.path = None

    @staticmethod
    def pending(dir_path: str) -> "list[str]":
        if not os.path.isdir(dir_path):
            return []
        return sorted(
            os.path.join(dir_path, name)
            for name in os.listdir(dir_path)
            if name.endswith(".jsonl")
        )

    @staticmethod
    def load(path: str) -> "list[dict]":
        # A line can be truncated mid-write if power was cut while writing
        # it; skip it rather than losing every turn before it.
        turns = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    turns.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return turns

    @staticmethod
    def delete(path: str) -> None:
        if os.path.exists(path):
            os.remove(path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_memory.py -v`
Expected: all tests PASS (existing `Memory` tests plus the new `RawLog` tests).

- [ ] **Step 5: Commit**

```bash
git add companion/memory.py tests/test_memory.py
git commit -m "feat: add RawLog for crash-safe incremental session transcripts"
```

---

### Task 2: `remember_session()` returns success/failure

**Files:**
- Modify: `companion/session.py:201-215` (`remember_session`)
- Test: `tests/test_session.py` (extend the existing `remember_session` test group, around line 203-237)

**Interfaces:**
- Consumes: nothing new.
- Produces: `remember_session(llm, memory, emit) -> bool` — `True` if there was nothing to save, or everything it tried to save succeeded; `False` if either the timeline append or the durable merge raised. Task 3 uses this return value to decide whether to delete the raw log.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_session.py`, directly after `test_remember_session_keeps_timeline_when_durable_fails` (around line 237):

```python
def test_remember_session_returns_true_on_success():
    llm = MagicMock()
    llm.has_user_turns.return_value = True
    llm.summarize.side_effect = ["- entry", "## Facts\n- x"]
    memory = MagicMock()
    memory.load_durable.return_value = ""
    assert session.remember_session(llm, memory, lambda e: None) is True


def test_remember_session_returns_true_without_user_turns():
    llm = MagicMock()
    llm.has_user_turns.return_value = False
    memory = MagicMock()
    assert session.remember_session(llm, memory, lambda e: None) is True


def test_remember_session_returns_false_when_timeline_fails():
    llm = MagicMock()
    llm.has_user_turns.return_value = True
    llm.summarize.side_effect = [Exception("blip"), "## Facts\n- x"]
    memory = MagicMock()
    memory.load_durable.return_value = ""
    assert session.remember_session(llm, memory, lambda e: None) is False


def test_remember_session_returns_false_when_durable_fails():
    llm = MagicMock()
    llm.has_user_turns.return_value = True
    llm.summarize.side_effect = ["- entry", Exception("blip")]
    memory = MagicMock()
    memory.load_durable.return_value = ""
    assert session.remember_session(llm, memory, lambda e: None) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_session.py -v -k remember_session_returns`
Expected: FAIL — `assert None is True` (current `remember_session` returns nothing).

- [ ] **Step 3: Implement the return value**

In `companion/session.py`, replace `remember_session` (lines 201-215):

```python
def remember_session(llm, memory, emit) -> bool:
    if not llm.has_user_turns():
        return True
    emit({"event": "system", "text": "Remembering this session..."})
    # Two independent side-channel calls: a failure in one must not skip the
    # other, and neither may crash the goodbye (mirrors the send/tts guards).
    ok = True
    try:
        memory.append_timeline(llm.summarize(config.TIMELINE_PROMPT))
    except Exception as exc:
        emit({"event": "warning", "text": f"Could not update timeline memory ({exc})."})
        ok = False
    try:
        merged = llm.summarize(config.DURABLE_MERGE_PROMPT + memory.load_durable())
        memory.write_durable(merged)
    except Exception as exc:
        emit({"event": "warning", "text": f"Could not update durable memory ({exc})."})
        ok = False
    return ok
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_session.py -v`
Expected: all tests PASS (existing `remember_session` tests are unaffected since they didn't check the return value; the four new ones pass).

- [ ] **Step 5: Commit**

```bash
git add companion/session.py tests/test_session.py
git commit -m "feat: remember_session reports whether the save succeeded"
```

---

### Task 3: wire `RawLog` into `run_loop`'s write path

**Files:**
- Modify: `companion/session.py` (import `RawLog`; `run_loop` signature and body, lines 218-275; `run_session`'s `run_loop` call site and try block, lines 126-184)
- Test: `tests/test_session.py` (`FakeRawLog` double, `run_scripted` helper, two existing `run_session` tests' fake loop signatures, new write-path tests)

**Interfaces:**
- Consumes: `RawLog` from Task 1 (`start()`, `append()`, `finish()`), `remember_session() -> bool` from Task 2.
- Produces: `run_loop(capture, transcriber, llm, memory, speaker, machine, raw_log, emit, should_stop) -> None` — new signature, `raw_log` inserted before `emit`. `run_session()` now constructs a `RawLog(config.RAW_LOG_DIR)` and passes it through. Task 4 adds a config value `RAW_LOG_DIR` and a recovery call that also uses this `raw_log`'s directory.

- [ ] **Step 1: Write the failing tests**

In `tests/test_session.py`, add `RAW_LOG_DIR` to config first isn't needed yet (Task 4). Add the `FakeRawLog` double directly after `FakeSpeaker` (around line 79):

```python
class FakeRawLog:
    def __init__(self):
        self.started = False
        self.appended = []
        self.finished = False

    def start(self):
        self.started = True
        self.finished = False

    def append(self, role, content):
        self.appended.append((role, content))

    def finish(self):
        self.finished = True
```

Update `run_scripted` (around line 82-104) to accept and wire in a `raw_log`:

```python
def run_scripted(texts, llm=None, speaker=None, memory=None, raw_log=None):
    """Drive run_loop through the scripted utterances, then stop."""
    events = []
    capture = FakeCapture(len(texts))
    transcriber = FakeTranscriber(texts)
    llm = llm if llm is not None else FakeLLM()
    speaker = speaker if speaker is not None else FakeSpeaker()
    raw_log = raw_log if raw_log is not None else FakeRawLog()
    if memory is None:
        memory = MagicMock()
        memory.load.return_value = "remembered stuff"
        memory.load_durable.return_value = ""
    session.run_loop(
        capture,
        transcriber,
        llm,
        memory,
        speaker,
        StateMachine(),
        raw_log,
        events.append,
        lambda: capture.exhausted,
    )
    return events, llm, speaker, memory
```

Add new tests directly after `test_forward_emits_heard_and_reply_and_speaks` (around line 138):

```python
def test_wake_starts_raw_log_and_appends_greeting():
    raw_log = FakeRawLog()
    run_scripted(["hey iris"], raw_log=raw_log)
    assert raw_log.started is True
    assert raw_log.appended == [("assistant", config.GREETING)]


def test_forward_appends_user_and_assistant_turns_to_raw_log():
    raw_log = FakeRawLog()
    run_scripted(["hey iris", "I like ramen"], raw_log=raw_log)
    assert ("user", "I like ramen") in raw_log.appended
    assert ("assistant", "Nice!") in raw_log.appended
```

Add new tests directly after `test_sleep_says_goodbye_and_remembers` (around line 163):

```python
def test_sleep_finishes_raw_log_when_remember_session_succeeds():
    raw_log = FakeRawLog()
    run_scripted(["hey iris", "I like ramen", "bye bye"], raw_log=raw_log)
    assert raw_log.finished is True


def test_sleep_keeps_raw_log_when_remember_session_fails():
    llm = FakeLLM()
    llm.summaries = [Exception("blip"), Exception("blip")]
    raw_log = FakeRawLog()
    run_scripted(["hey iris", "I like ramen", "bye bye"], llm=llm, raw_log=raw_log)
    assert raw_log.finished is False
```

Add a new test directly after `test_stop_while_awake_runs_the_goodbye_path` (around line 171):

```python
def test_stop_while_awake_finishes_raw_log_on_successful_flush():
    raw_log = FakeRawLog()
    run_scripted(["hey iris", "I like ramen"], raw_log=raw_log)
    assert raw_log.finished is True
```

Update `test_run_session_happy_path_runs_loop_and_closes_capture`'s `fake_loop` (around line 323) to accept the new parameter:

```python
    def fake_loop(capture, transcriber, llm, memory, speaker, machine, raw_log, emit, should_stop):
        loop_ran["done"] = True
```

(`test_run_session_emits_error_when_loop_raises`'s `exploding_loop(*args, **kwargs)` already accepts any signature — no change needed there.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_session.py -v`
Expected: FAIL — `TypeError: run_loop() missing 1 required positional argument: 'raw_log'` (or similar) since `run_loop` doesn't accept `raw_log` yet, and `fake_loop` now has a mismatched signature vs. the still-old call site.

- [ ] **Step 3: Implement the write path**

In `companion/session.py`, update the import line near the top:

```python
from companion.memory import Memory, RawLog
```

Replace `run_loop`'s signature and body (lines 218-275) with:

```python
def run_loop(capture, transcriber, llm, memory, speaker, machine, raw_log, emit, should_stop) -> None:
    while not should_stop():
        emit({"event": "status", "state": "listening"})
        audio = capture.listen_for_utterance(stop_check=should_stop)
        if should_stop():
            break
        if audio is None or len(audio) == 0:
            continue
        emit({"event": "status", "state": "thinking"})
        try:
            text = transcriber.transcribe(audio)
        except Exception as exc:
            # Cloud STT can fail on a network blip; a local frame can be bad.
            # Drop this utterance and keep listening instead of crashing.
            emit({"event": "warning", "text": f"Transcription failed ({exc})."})
            continue
        if not text:
            continue

        action = machine.process(text)
        if action == Action.IGNORE:
            continue
        elif action == Action.WAKE:
            emit({"event": "system", "text": "Waking up."})
            # Fresh history per session, and the greeting is seeded as an
            # assistant turn — otherwise the LLM doesn't know the opening
            # question was already asked and re-asks it.
            llm.reset(memory.load())
            llm.seed_assistant(config.GREETING)
            raw_log.start()
            raw_log.append("assistant", config.GREETING)
            emit({"event": "reply", "text": config.GREETING})
            _speak(speaker, config.GREETING, emit)
        elif action == Action.CANCEL:
            emit({"event": "system", "text": "Discarded that."})
        elif action == Action.SLEEP:
            emit({"event": "system", "text": "Going back to sleep."})
            _speak(speaker, "Bye for now!", emit)
            if remember_session(llm, memory, emit):
                raw_log.finish()
        elif action == Action.FORWARD:
            emit({"event": "heard", "text": text})
            try:
                reply = llm.send(text)
            except Exception as exc:
                # Cloud APIs hiccup and Ollama can die mid-session; keep the
                # session alive. (The dangling user turn is harmless: every
                # provider accepts consecutive user messages.)
                emit({"event": "warning", "text": f"The brain failed to reply ({exc})."})
                _speak(speaker, "Sorry, I had trouble thinking. Say that again?", emit)
                continue
            raw_log.append("user", text)
            raw_log.append("assistant", reply)
            emit({"event": "reply", "text": reply})
            _speak(speaker, reply, emit)

    # Stopped by flag (End button / Ctrl+C path) while a conversation was
    # active: run the same goodbye as the stop phrase so memory is never
    # silently dropped.
    if machine.state == State.ACTIVE:
        _speak(speaker, "Bye for now!", emit)
        if remember_session(llm, memory, emit):
            raw_log.finish()
```

In `companion/config.py`, add `import os` as the first line of the file (it currently has no imports), then add this near `MEMORY_DIR`:

```python
RAW_LOG_DIR = os.path.join(MEMORY_DIR, "raw")
```

In `run_session`, add `raw_log` construction to the try block (after the `memory = Memory(...)` line, around line 141):

```python
        memory = Memory(config.MEMORY_DIR, config.TIMELINE_MAX_CHARS)
        raw_log = RawLog(config.RAW_LOG_DIR)
```

And update the `run_loop(...)` call site in `run_session` (lines 165-174):

```python
        run_loop(
            capture,
            transcriber,
            llm,
            memory,
            speaker,
            StateMachine(),
            raw_log,
            emit,
            should_stop,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_session.py tests/test_main.py tests/test_server.py -v`
Expected: all tests PASS. (`test_main.py`/`test_server.py` don't touch `run_loop` directly but exercise `run_session`/`server.py` code paths — confirm nothing else broke.)

- [ ] **Step 5: Commit**

```bash
git add companion/config.py companion/session.py tests/test_session.py
git commit -m "feat: write conversation turns to a crash-safe raw log during the session"
```

---

### Task 4: startup recovery of orphaned sessions

**Files:**
- Modify: `companion/session.py` (new `recover_orphaned_sessions` function; wire into `run_session`)
- Test: `tests/test_session.py` (new `FakeProvider` double, unit tests for `recover_orphaned_sessions`, integration tests via `run_session`)

**Interfaces:**
- Consumes: `RawLog.pending`/`.load`/`.delete` (Task 1), `remember_session() -> bool` (Task 2), `config.RAW_LOG_DIR` (added in Task 3).
- Produces: `recover_orphaned_sessions(provider, memory, raw_log_dir, emit) -> None`, called once from `run_session()` before the "Ready" message.

- [ ] **Step 1: Write the failing tests**

Add near the top of `tests/test_session.py`, after the `FakeSpeaker`/`FakeRawLog` doubles:

```python
class FakeProvider:
    """Records chat() calls; returns queued replies (default 'OK.')."""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = []

    def chat(self, system, turns):
        self.calls.append((system, [dict(turn) for turn in turns]))
        return self.replies.pop(0) if self.replies else "OK."
```

Add unit tests for `recover_orphaned_sessions` directly after the `remember_session` test group (after `test_remember_session_returns_false_when_durable_fails` from Task 2):

```python
def test_recover_orphaned_sessions_does_nothing_when_dir_missing(tmp_path):
    memory = MagicMock()
    events = []

    session.recover_orphaned_sessions(
        object(), memory, str(tmp_path / "missing"), events.append
    )

    assert events == []
    memory.append_timeline.assert_not_called()


def test_recover_orphaned_sessions_deletes_greeting_only_file_without_summarizing(tmp_path):
    # Crash right after wake, before the user said anything: nothing to
    # summarize, but the orphan file should still be cleaned up.
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    orphan = raw_dir / "20260709-090000.jsonl"
    orphan.write_text(
        '{"role": "assistant", "content": "Hey Caio, good to hear you!"}\n',
        encoding="utf-8",
    )
    memory = MagicMock()

    session.recover_orphaned_sessions(object(), memory, str(raw_dir), lambda e: None)

    assert not orphan.exists()
    memory.append_timeline.assert_not_called()
    memory.write_durable.assert_not_called()


def test_recover_orphaned_sessions_processes_oldest_first_and_deletes_on_success(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "20260709-090000.jsonl").write_text(
        '{"role": "user", "content": "first"}\n', encoding="utf-8"
    )
    (raw_dir / "20260710-090000.jsonl").write_text(
        '{"role": "user", "content": "second"}\n', encoding="utf-8"
    )
    memory = MagicMock()
    memory.load_durable.return_value = ""
    provider = FakeProvider(["- a", "## Facts\n- a", "- b", "## Facts\n- b"])
    events = []

    session.recover_orphaned_sessions(provider, memory, str(raw_dir), events.append)

    assert list(raw_dir.iterdir()) == []
    assert memory.append_timeline.call_count == 2
    assert memory.write_durable.call_count == 2
    # First provider call is for the older ("first") orphan.
    assert provider.calls[0][1][0] == {"role": "user", "content": "first"}
    assert any(e["event"] == "system" and "2 session" in e["text"] for e in events)


def test_recover_orphaned_sessions_keeps_file_when_save_fails(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    orphan = raw_dir / "20260709-090000.jsonl"
    orphan.write_text('{"role": "user", "content": "hi"}\n', encoding="utf-8")
    memory = MagicMock()
    memory.load_durable.return_value = ""

    class BoomProvider:
        def chat(self, system, turns):
            raise RuntimeError("api down")

    session.recover_orphaned_sessions(BoomProvider(), memory, str(raw_dir), lambda e: None)

    assert orphan.exists()
```

Add integration tests directly after `test_run_session_happy_path_runs_loop_and_closes_capture`:

```python
def test_run_session_recovers_orphaned_raw_log_before_starting(monkeypatch, tmp_path):
    memory_dir = tmp_path / "memory"
    raw_dir = memory_dir / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "20260709-230000.jsonl").write_text(
        '{"role": "assistant", "content": "Hey Caio, good to hear you!"}\n'
        '{"role": "user", "content": "I visited Rome!"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "MEMORY_DIR", str(memory_dir))
    monkeypatch.setattr(config, "RAW_LOG_DIR", str(raw_dir))
    monkeypatch.setattr(session, "preflight_error", lambda b, e: None)
    monkeypatch.setattr(session, "build_capture", lambda ptt: MagicMock())
    monkeypatch.setattr(session, "load_transcriber", lambda ears: object())
    monkeypatch.setattr(
        session,
        "make_provider",
        lambda brain: FakeProvider(["- Talked about Rome.", "## Facts\n- Visited Rome."]),
    )
    monkeypatch.setattr(session, "Speaker", lambda *args: FakeSpeaker())
    monkeypatch.setattr(session, "run_loop", lambda *args, **kwargs: None)

    events = []
    ok = session.run_session("claude", "local", False, events.append, lambda: True)

    assert ok is True
    assert not (raw_dir / "20260709-230000.jsonl").exists()
    assert (memory_dir / "timeline.md").exists()
    assert (memory_dir / "durable.md").exists()
    system_lines = [e["text"] for e in events if e["event"] == "system"]
    assert any("Recovering 1 session" in t for t in system_lines)


def test_run_session_skips_recovery_when_no_orphans(monkeypatch, tmp_path):
    memory_dir = tmp_path / "memory"
    monkeypatch.setattr(config, "MEMORY_DIR", str(memory_dir))
    monkeypatch.setattr(config, "RAW_LOG_DIR", str(memory_dir / "raw"))
    monkeypatch.setattr(session, "preflight_error", lambda b, e: None)
    monkeypatch.setattr(session, "build_capture", lambda ptt: MagicMock())
    monkeypatch.setattr(session, "load_transcriber", lambda ears: object())
    monkeypatch.setattr(session, "make_provider", lambda brain: object())
    monkeypatch.setattr(session, "Speaker", lambda *args: FakeSpeaker())
    monkeypatch.setattr(session, "run_loop", lambda *args, **kwargs: None)

    events = []
    session.run_session("claude", "local", False, events.append, lambda: True)

    assert not any(
        "Recovering" in e["text"] for e in events if e["event"] == "system"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_session.py -v -k recover`
Expected: FAIL — `AttributeError: module 'companion.session' has no attribute 'recover_orphaned_sessions'`.

- [ ] **Step 3: Implement recovery**

In `companion/session.py`, add this function directly after `remember_session` (after Task 2's version, before `run_loop`):

```python
def recover_orphaned_sessions(provider, memory, raw_log_dir, emit) -> None:
    """Fold any session that never got a proper goodbye (crash, power loss)
    into memory, the same way remember_session() does at a normal sleep."""
    paths = RawLog.pending(raw_log_dir)
    if not paths:
        return
    emit(
        {
            "event": "system",
            "text": f"Recovering {len(paths)} session(s) that weren't saved last time...",
        }
    )
    for path in paths:
        recovery_llm = LLMClient(provider, config.SYSTEM_PROMPT)
        recovery_llm.turns = RawLog.load(path)
        if remember_session(recovery_llm, memory, emit):
            RawLog.delete(path)
```

In `run_session`, call it right after constructing `raw_log` (added in Task 3), before `speaker` is constructed:

```python
        memory = Memory(config.MEMORY_DIR, config.TIMELINE_MAX_CHARS)
        raw_log = RawLog(config.RAW_LOG_DIR)
        recover_orphaned_sessions(llm.provider, memory, config.RAW_LOG_DIR, emit)
        speaker = Speaker(
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_session.py -v`
Expected: all tests PASS.

Run the full suite to confirm nothing else broke:

Run: `python -m pytest -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add companion/session.py tests/test_session.py
git commit -m "feat: recover orphaned session transcripts into memory on startup"
```

---

## Manual verification (after all tasks)

1. Start Iris (`python -m companion.main` or the web UI), say the wake phrase, say something, then kill the process hard (Task Manager / `taskkill /F`, not a graceful Ctrl+C) — simulates the power-loss scenario.
2. Confirm a file exists under `memory/raw/`.
3. Restart Iris. Confirm a system message like "Recovering 1 session(s)..." appears, `memory/raw/` is empty afterward, and `memory/timeline.md`/`memory/durable.md` reflect what was said before the kill.
