# Thinking Room + No Stage Directions (v1.2.1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the user 2 seconds of thinking silence before the app responds, and guarantee stage directions like "(laughs)" are never spoken.

**Architecture:** One config constant change (`SILENCE_TIMEOUT_MS`), one `SYSTEM_PROMPT` clause rewrite, and a `_strip_stage_directions()` helper applied inside `LLMClient.send()` before the reply is stored in history or returned.

**Tech Stack:** Python 3.13, pytest. Spec: `docs/superpowers/specs/2026-07-08-thinking-room-design.md`.

## Global Constraints

- Run tests with `python -m pytest` (bare `pytest` is not on PATH on this machine).
- Replies must be cleaned BEFORE entering `self.history`.
- All 28 existing tests must keep passing after every task.

---

### Task 1: Strip stage directions in LLMClient

**Files:**
- Modify: `companion/llm_client.py`
- Test: `tests/test_llm_client.py` (append new tests)

**Interfaces:**
- Produces: module-private `_strip_stage_directions(text: str) -> str`; `send()` behavior change (cleaned replies). Public signatures unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm_client.py`:

```python
def test_send_strips_stage_directions_before_storing_and_returning():
    fake_response = {
        "message": {
            "role": "assistant",
            "content": "(laughs) Well, Rome sounds *smiling* amazing (pauses) .",
        }
    }
    with patch("companion.llm_client.ollama.chat", return_value=fake_response):
        client = LLMClient("llama3.1:8b", "system prompt")
        reply = client.send("I visited Rome!")

    assert reply == "Well, Rome sounds amazing."
    assert client.history[-1] == {"role": "assistant", "content": "Well, Rome sounds amazing."}


def test_send_leaves_clean_replies_untouched():
    fake_response = {"message": {"role": "assistant", "content": "Nice! How was Rome?"}}
    with patch("companion.llm_client.ollama.chat", return_value=fake_response):
        client = LLMClient("llama3.1:8b", "system prompt")
        reply = client.send("I visited Rome!")

    assert reply == "Nice! How was Rome?"
```

- [ ] **Step 2: Run tests to verify the new one fails**

Run: `python -m pytest tests/test_llm_client.py -v`
Expected: `test_send_strips_stage_directions_before_storing_and_returning` FAILS (reply still contains "(laughs)"); all others PASS.

- [ ] **Step 3: Implement**

In `companion/llm_client.py`, add at the top (below `import ollama`):

```python
import re

# llama3.1 writes roleplay stage directions — "(laughs)", "*smiles*" —
# even when the system prompt forbids them, and the TTS would read them
# aloud verbatim. Strip them before the reply is stored or spoken; keeping
# them out of history also stops the model imitating its own habit.
_STAGE_DIRECTIONS = re.compile(r"\([^)]*\)|\*[^*]*\*")


def _strip_stage_directions(text: str) -> str:
    cleaned = _STAGE_DIRECTIONS.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    return cleaned.strip()
```

(Move `import re` above `import ollama` to keep stdlib imports first.)

In `send()`, replace:

```python
        reply = response["message"]["content"]
```

with:

```python
        reply = _strip_stage_directions(response["message"]["content"])
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: 30 passed.

- [ ] **Step 5: Commit**

```bash
git add companion/llm_client.py tests/test_llm_client.py
git commit -m "fix: strip roleplay stage directions from LLM replies"
```

---

### Task 2: Config — thinking pause + prompt hardening

**Files:**
- Modify: `companion/config.py`

**Interfaces:**
- Consumes: nothing from Task 1. `VoiceDetector` already reads `SILENCE_TIMEOUT_MS`.

- [ ] **Step 1: Lengthen the silence timeout**

In `companion/config.py`, replace:

```python
SILENCE_TIMEOUT_MS = 800
```

with:

```python
# 2s so a language learner can pause to think mid-sentence without being
# cut off; also the delay before the companion starts answering.
SILENCE_TIMEOUT_MS = 2000
```

- [ ] **Step 2: Harden the prompt**

In `SYSTEM_PROMPT`, replace the final clause:

```python
    "spoken aloud, not read: keep them short (one to three sentences), "
    "natural, and free of lists, markdown, emojis, and stage directions."
```

with:

```python
    "spoken aloud, not read: keep them short (one to three sentences), "
    "natural, and free of lists, markdown, and emojis. Never write "
    "actions, emotions, or sounds in parentheses or asterisks — no "
    '"(laughs)", "(smiling)", "*pauses*" — only the exact words you '
    "would speak out loud."
)
```

(The closing parenthesis of the assignment stays as-is; only the quoted
clause changes.)

- [ ] **Step 3: Verify**

Run: `python -m pytest -q` — expected: 30 passed.
Run: `python -c "from companion import config; assert config.SILENCE_TIMEOUT_MS == 2000; assert 'laughs' in config.SYSTEM_PROMPT; print('CONFIG OK')"` — expected: `CONFIG OK`.

- [ ] **Step 4: Commit**

```bash
git add companion/config.py
git commit -m "feat: 2s thinking pause and explicit stage-direction ban"
```

---

## Final verification (after Task 2)

1. `python -m pytest -q` → 30 passed.
2. Manual (user): pause ~1.5 s mid-sentence without being cut off; confirm
   no spoken "(laughs)"-style artifacts across a few sessions.
