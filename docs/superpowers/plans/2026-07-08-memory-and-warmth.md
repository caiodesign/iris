# Memory + Warmth (v1.2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the companion persistent memory (LLM-written session summaries in `memory.md`) and a warmer "curious friend" personality that knows the user's name.

**Architecture:** New pure-file-I/O module `companion/memory.py` (load/append). `LLMClient` grows `reset(memory)`, `summarize(instruction)`, and `has_user_turns()` — no config imports there; `main.py` wires memory load at WAKE and summary write at SLEEP. Prompt/name/summary-instruction changes are config-only.

**Tech Stack:** Python 3.13 on Windows 11, Ollama (llama3.1:8b), pytest. Spec: `docs/superpowers/specs/2026-07-08-memory-and-warmth-design.md`.

## Global Constraints

- Run tests with `python -m pytest` (bare `pytest` is not on PATH on this machine).
- Raw transcripts are never persisted — only LLM-written summaries.
- A failed summary write must never crash the app (WARN and continue).
- `memory.md` is git-ignored (personal data).
- All 21 existing tests must keep passing after every task.

---

### Task 1: Memory module

**Files:**
- Create: `companion/memory.py`
- Test: `tests/test_memory.py` (new file)

**Interfaces:**
- Produces: `Memory(path: str, max_chars: int)` with `load() -> str` (tail of file up to `max_chars`, `""` if missing) and `append_session(summary: str) -> None` (appends `## YYYY-MM-DD HH:MM` heading + summary + blank line).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_memory.py` with:

```python
# tests/test_memory.py
from companion.memory import Memory


def test_load_returns_empty_string_when_file_missing(tmp_path):
    memory = Memory(str(tmp_path / "memory.md"), 6000)
    assert memory.load() == ""


def test_append_session_writes_dated_heading_and_summary(tmp_path):
    path = tmp_path / "memory.md"
    memory = Memory(str(path), 6000)

    memory.append_session("- Talked about food.\n- Caio visited Japan.")

    content = path.read_text(encoding="utf-8")
    assert content.startswith("## 20")  # "## 2026-07-08 14:30" style heading
    assert "- Talked about food.\n- Caio visited Japan." in content


def test_append_session_accumulates_sessions(tmp_path):
    path = tmp_path / "memory.md"
    memory = Memory(str(path), 6000)

    memory.append_session("- First session.")
    memory.append_session("- Second session.")

    content = path.read_text(encoding="utf-8")
    assert "- First session." in content
    assert "- Second session." in content
    assert content.count("## 20") == 2


def test_load_returns_at_most_max_chars_keeping_the_end(tmp_path):
    path = tmp_path / "memory.md"
    memory = Memory(str(path), 20)

    memory.append_session("A" * 30)
    loaded = memory.load()

    assert len(loaded) <= 20
    assert loaded == "A" * 20  # the END of the file survives truncation
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_memory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'companion.memory'`.

- [ ] **Step 3: Write the implementation**

Create `companion/memory.py`:

```python
# companion/memory.py
import os
from datetime import datetime


class Memory:
    def __init__(self, path: str, max_chars: int):
        self.path = path
        self.max_chars = max_chars

    def load(self) -> str:
        if not os.path.exists(self.path):
            return ""
        with open(self.path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return text[-self.max_chars :]

    def append_session(self, summary: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"## {stamp}\n{summary.strip()}\n\n")
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: 25 passed.

- [ ] **Step 5: Commit**

```bash
git add companion/memory.py tests/test_memory.py
git commit -m "feat: add Memory module for session summaries"
```

---

### Task 2: LLMClient memory hooks

**Files:**
- Modify: `companion/llm_client.py`
- Test: `tests/test_llm_client.py` (append new tests; keep the existing four)

**Interfaces:**
- Produces: `LLMClient.reset(memory: str = "") -> None`, `LLMClient.summarize(instruction: str) -> str` (does NOT mutate history), `LLMClient.has_user_turns() -> bool`.
- Consumes: nothing from Task 1 (modules stay decoupled; `main.py` joins them in Task 3).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm_client.py`:

```python
def test_reset_with_memory_injects_remembered_context():
    client = LLMClient("llama3.1:8b", "Base prompt.")
    client.reset("- Caio visited Japan.")

    assert client.history == [
        {
            "role": "system",
            "content": (
                "Base prompt.\n\nWhat you remember about the user from "
                "previous sessions:\n- Caio visited Japan."
            ),
        }
    ]


def test_summarize_asks_ollama_without_mutating_history():
    fake_response = {"message": {"role": "assistant", "content": "- Bullets."}}
    with patch("companion.llm_client.ollama.chat", return_value=fake_response) as mock_chat:
        client = LLMClient("llama3.1:8b", "system prompt")
        client.seed_assistant("Hi!")
        history_before = list(client.history)

        result = client.summarize("Summarize the session.")

    assert result == "- Bullets."
    assert client.history == history_before
    mock_chat.assert_called_once_with(
        model="llama3.1:8b",
        messages=history_before + [{"role": "user", "content": "Summarize the session."}],
    )


def test_has_user_turns_false_for_fresh_session_true_after_send():
    fake_response = {"message": {"role": "assistant", "content": "Sure!"}}
    with patch("companion.llm_client.ollama.chat", return_value=fake_response):
        client = LLMClient("llama3.1:8b", "system prompt")
        client.seed_assistant("Hi!")
        assert client.has_user_turns() is False

        client.send("Hello")
        assert client.has_user_turns() is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_llm_client.py -v`
Expected: the three new tests FAIL (`TypeError: reset() takes 1 positional argument` / `AttributeError: 'LLMClient' object has no attribute 'summarize'` / `... 'has_user_turns'`); the existing four PASS.

- [ ] **Step 3: Implement**

In `companion/llm_client.py`, replace the `reset` method and add the two new methods after it:

```python
    def reset(self, memory: str = "") -> None:
        content = self.system_prompt
        if memory:
            content += (
                "\n\nWhat you remember about the user from previous "
                "sessions:\n" + memory
            )
        self.history = [{"role": "system", "content": content}]

    def summarize(self, instruction: str) -> str:
        # The session is over: the instruction and reply deliberately stay
        # out of self.history — this is a side-channel request.
        messages = list(self.history) + [{"role": "user", "content": instruction}]
        response = ollama.chat(model=self.model, messages=messages)
        return response["message"]["content"]

    def has_user_turns(self) -> bool:
        return any(message["role"] == "user" for message in self.history)
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: 28 passed.

- [ ] **Step 5: Commit**

```bash
git add companion/llm_client.py tests/test_llm_client.py
git commit -m "feat: add memory-aware reset, summarize, has_user_turns to LLMClient"
```

---

### Task 3: Config, main wiring, warm prompt

**Files:**
- Modify: `companion/config.py`
- Modify: `companion/main.py`
- Modify: `.gitignore` (add `memory.md`)
- Modify: `README.md` (usage note)

**Interfaces:**
- Consumes: `Memory(path, max_chars)` / `load()` / `append_session(summary)` from Task 1; `reset(memory)`, `summarize(instruction)`, `has_user_turns()` from Task 2.
- Produces: config constants `USER_NAME`, `MEMORY_PATH`, `MEMORY_MAX_CHARS`, `SUMMARY_PROMPT`; rewritten `SYSTEM_PROMPT` and `GREETING`.

- [ ] **Step 1: Config changes**

In `companion/config.py`, add directly above the `GREETING` line:

```python
USER_NAME = "Caio"

MEMORY_PATH = "memory.md"
MEMORY_MAX_CHARS = 6000

SUMMARY_PROMPT = (
    "The session is over. Summarize it in 3 to 5 short bullet points for "
    "your own memory before the next session: topics discussed, English "
    "mistakes the user made, and personal facts you learned about the user "
    "(trips, food, family, work, plans). Write only the bullet points."
)
```

Then replace the `GREETING` and `SYSTEM_PROMPT` assignments with:

```python
GREETING = f"Hey {USER_NAME}, good to hear you! So — what are we diving into today?"

SYSTEM_PROMPT = (
    f"You are Chat, a voice companion helping {USER_NAME} practice English "
    "conversation. You are a warm, curious friend: caring, genuinely "
    "interested in his life, playful but never over the top. Use his name "
    "naturally, the way a friend does — sometimes, not constantly. React "
    "to what he says the way a good friend would — surprise, delight, a "
    "little gentle teasing — instead of just answering. Ask follow-up "
    "questions about things he mentions, and when you remember something "
    "from a previous session, bring it up yourself (for example, ask how "
    "that trip he mentioned went) instead of waiting for him to repeat "
    "it. Give encouragement only when he's earned it, so it means "
    "something. At the very start of a session, ask what he'd like to "
    "focus on today as an open question (for example: free conversation, "
    "grammar correction, or vocabulary building) rather than reading a "
    "fixed menu, then adapt your style to his answer. Your replies are "
    "spoken aloud, not read: keep them short (one to three sentences), "
    "natural, and free of lists, markdown, emojis, and stage directions."
)
```

- [ ] **Step 2: Wire memory into main**

In `companion/main.py`:

a. Add the import below the existing `from companion.llm_client import LLMClient` line:

```python
from companion.memory import Memory
```

b. In `main()`, after `llm = LLMClient(config.OLLAMA_MODEL, config.SYSTEM_PROMPT)`, add:

```python
    memory = Memory(config.MEMORY_PATH, config.MEMORY_MAX_CHARS)
```

c. In the WAKE branch, replace `llm.reset()` with:

```python
                llm.reset(memory.load())
```

(keep the existing comment and the `seed_assistant`/`speak_safely` lines as they are).

d. Replace the whole SLEEP branch with:

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

- [ ] **Step 3: Hygiene**

Add `memory.md` on its own line to `.gitignore`.

In `README.md`, append to the "Usage notes" section:

```markdown
- The companion keeps a memory: when you say "Bye Bye" it writes a short
  summary of the session to `memory.md`, and reads it back at the next
  "Hey Chat". Open or edit that file any time to see or change what it
  remembers; delete it to make the companion forget everything.
```

- [ ] **Step 4: Verify**

Run: `python -m pytest -q` — expected: 28 passed.
Run: `python -c "import companion.main"` — expected: exit 0.
Run: `python -c "from companion import config; print(config.GREETING); print(config.SUMMARY_PROMPT[:40])"` — expected: greeting contains "Caio".

- [ ] **Step 5: Commit**

```bash
git add companion/config.py companion/main.py .gitignore README.md
git commit -m "feat: wire session memory into main and warm up the personality"
```

---

## Final verification (after Task 3)

1. `python -m pytest -q` → 28 passed.
2. Manual acceptance (user): have a session mentioning something personal,
   say "bye bye", open `memory.md` and check the summary; say "hey chat"
   again and see whether the companion brings it up.
