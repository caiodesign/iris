# Switchable Brains (v1.3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user pick the conversation brain at launch — local llama (default, free) or Claude / OpenAI / z.ai cloud APIs — with keys in a git-ignored `.env`.

**Architecture:** New `companion/providers.py` holds one adapter per provider behind a duck-typed `chat(system, turns) -> str` interface. `LLMClient` is refactored to store the system prompt and turns separately and delegate to an injected provider (public interface unchanged). `main.py` gains `--brain`, an interactive menu, `.env` loading, provider-aware startup checks, and a guarded `send`.

**Tech Stack:** Python 3.13, `anthropic` SDK, `openai` SDK (also serves z.ai's OpenAI-compatible endpoint `https://api.z.ai/api/paas/v4/`), `python-dotenv`, pytest. Spec: `docs/superpowers/specs/2026-07-08-cloud-brains-design.md`.

## Global Constraints

- Run tests with `python -m pytest` (bare `pytest` is not on PATH on this machine).
- Models: `ANTHROPIC_MODEL = "claude-sonnet-5"`, `OPENAI_MODEL = "gpt-5.4"`, `ZAI_MODEL = "glm-5"`, `CLOUD_MAX_TOKENS = 1024`. No sampling params, no thinking config on Claude calls.
- Keys only via env vars loaded from `.env` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `ZAI_API_KEY`); never in code or git.
- `LLMClient` public interface must not change: `send`, `reset(memory="")`, `seed_assistant`, `summarize(instruction)`, `has_user_turns`.
- All 30 existing tests must keep passing after every task (llm_client tests get rewritten in Task 2, same coverage).

---

### Task 1: Providers module

**Files:**
- Modify: `requirements.txt`
- Modify: `companion/config.py` (provider constants)
- Create: `companion/providers.py`
- Test: `tests/test_providers.py` (new file)

**Interfaces:**
- Produces: `LocalProvider(model)`, `ClaudeProvider(model, max_tokens)`, `OpenAICompatProvider(model, api_key_env, base_url=None)` — each with `chat(system: str, turns: list[dict]) -> str`; `make_provider(name: str)`; `REQUIRED_ENV: dict[str, str]` (keys `"claude"|"openai"|"zai"`); config constants `LLM_PROVIDER`, `ANTHROPIC_MODEL`, `OPENAI_MODEL`, `ZAI_MODEL`, `ZAI_BASE_URL`, `CLOUD_MAX_TOKENS`.

- [ ] **Step 1: Add dependencies and config**

Append to `requirements.txt`:

```
anthropic>=0.60,<1
openai>=1.50,<3
python-dotenv>=1.0,<2
```

Run: `pip install "anthropic>=0.60,<1" "openai>=1.50,<3" "python-dotenv>=1.0,<2"`

In `companion/config.py`, add directly below the `OLLAMA_MODEL` line:

```python
# Which brain answers: "local" (Ollama, free), "claude", "openai", or "zai".
# Pick at launch via the startup menu or --brain; this is the Enter default.
LLM_PROVIDER = "local"

ANTHROPIC_MODEL = "claude-sonnet-5"
OPENAI_MODEL = "gpt-5.4"
ZAI_MODEL = "glm-5"
ZAI_BASE_URL = "https://api.z.ai/api/paas/v4/"
CLOUD_MAX_TOKENS = 1024
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_providers.py`:

```python
# tests/test_providers.py
from unittest.mock import MagicMock, patch

import pytest

from companion.providers import (
    REQUIRED_ENV,
    ClaudeProvider,
    LocalProvider,
    OpenAICompatProvider,
    make_provider,
)


def test_required_env_covers_all_cloud_providers():
    assert REQUIRED_ENV == {
        "claude": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "zai": "ZAI_API_KEY",
    }


def test_local_provider_prepends_system_message():
    fake_response = {"message": {"role": "assistant", "content": "Hi!"}}
    with patch("companion.providers.ollama.chat", return_value=fake_response) as mock_chat:
        provider = LocalProvider("llama3.1:8b")
        reply = provider.chat("Be nice.", [{"role": "user", "content": "Hello"}])

    assert reply == "Hi!"
    mock_chat.assert_called_once_with(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": "Be nice."},
            {"role": "user", "content": "Hello"},
        ],
    )


def _fake_claude_response(text="Hi Caio!", stop_reason="end_turn"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.stop_reason = stop_reason
    return response


def test_claude_provider_passes_system_and_turns():
    with patch("companion.providers.anthropic.Anthropic") as MockAnthropic:
        client = MockAnthropic.return_value
        client.messages.create.return_value = _fake_claude_response()

        provider = ClaudeProvider("claude-sonnet-5", 1024)
        reply = provider.chat("Be nice.", [{"role": "user", "content": "Hello"}])

    assert reply == "Hi Caio!"
    client.messages.create.assert_called_once_with(
        model="claude-sonnet-5",
        max_tokens=1024,
        system="Be nice.",
        messages=[{"role": "user", "content": "Hello"}],
    )


def test_claude_provider_folds_leading_assistant_greeting_into_system():
    # Claude's API requires the first message to be a user turn, but our
    # sessions start with the seeded assistant greeting.
    with patch("companion.providers.anthropic.Anthropic") as MockAnthropic:
        client = MockAnthropic.return_value
        client.messages.create.return_value = _fake_claude_response()

        provider = ClaudeProvider("claude-sonnet-5", 1024)
        provider.chat(
            "Be nice.",
            [
                {"role": "assistant", "content": "Hey Caio!"},
                {"role": "user", "content": "Hello"},
            ],
        )

    kwargs = client.messages.create.call_args.kwargs
    assert 'You already opened this session by saying: "Hey Caio!"' in kwargs["system"]
    assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]


def test_claude_provider_raises_on_refusal():
    with patch("companion.providers.anthropic.Anthropic") as MockAnthropic:
        client = MockAnthropic.return_value
        client.messages.create.return_value = _fake_claude_response(stop_reason="refusal")

        provider = ClaudeProvider("claude-sonnet-5", 1024)
        with pytest.raises(RuntimeError):
            provider.chat("Be nice.", [{"role": "user", "content": "Hello"}])


def test_openai_compat_provider_wires_key_base_url_and_extracts_reply(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="Oi!"))]

    with patch("companion.providers.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = fake_response

        provider = OpenAICompatProvider(
            "glm-5", "ZAI_API_KEY", base_url="https://api.z.ai/api/paas/v4/"
        )
        reply = provider.chat("Be nice.", [{"role": "user", "content": "Hello"}])

    assert reply == "Oi!"
    MockOpenAI.assert_called_once_with(
        api_key="test-key", base_url="https://api.z.ai/api/paas/v4/"
    )
    MockOpenAI.return_value.chat.completions.create.assert_called_once_with(
        model="glm-5",
        messages=[
            {"role": "system", "content": "Be nice."},
            {"role": "user", "content": "Hello"},
        ],
    )


def test_make_provider_rejects_unknown_name():
    with pytest.raises(ValueError):
        make_provider("skynet")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_providers.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'companion.providers'`.

- [ ] **Step 4: Implement**

Create `companion/providers.py`:

```python
# companion/providers.py
import os

import anthropic
import ollama
from openai import OpenAI

# Which environment variable each cloud provider needs ("local" needs none).
REQUIRED_ENV = {
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "zai": "ZAI_API_KEY",
}


class LocalProvider:
    def __init__(self, model: str):
        self.model = model

    def chat(self, system: str, turns: list) -> str:
        messages = [{"role": "system", "content": system}] + turns
        response = ollama.chat(model=self.model, messages=messages)
        return response["message"]["content"]


class ClaudeProvider:
    def __init__(self, model: str, max_tokens: int):
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.model = model
        self.max_tokens = max_tokens

    def chat(self, system: str, turns: list) -> str:
        # Claude's API requires the first message to be a user turn, but our
        # sessions start with the seeded assistant greeting — fold any leading
        # assistant turns into the system text instead of sending them.
        turns = list(turns)
        while turns and turns[0]["role"] == "assistant":
            opener = turns.pop(0)
            system += (
                "\n\nYou already opened this session by saying: "
                f"\"{opener['content']}\""
            )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=turns,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("Claude declined to answer this request.")
        return " ".join(
            block.text for block in response.content if block.type == "text"
        ).strip()


class OpenAICompatProvider:
    def __init__(self, model: str, api_key_env: str, base_url: str = None):
        self.client = OpenAI(api_key=os.environ[api_key_env], base_url=base_url)
        self.model = model

    def chat(self, system: str, turns: list) -> str:
        messages = [{"role": "system", "content": system}] + turns
        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        return response.choices[0].message.content


def make_provider(name: str):
    from companion import config

    if name == "local":
        return LocalProvider(config.OLLAMA_MODEL)
    if name == "claude":
        return ClaudeProvider(config.ANTHROPIC_MODEL, config.CLOUD_MAX_TOKENS)
    if name == "openai":
        return OpenAICompatProvider(config.OPENAI_MODEL, "OPENAI_API_KEY")
    if name == "zai":
        return OpenAICompatProvider(
            config.ZAI_MODEL, "ZAI_API_KEY", base_url=config.ZAI_BASE_URL
        )
    raise ValueError(f"Unknown provider: {name}")
```

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: 37 passed (30 existing + 7 new).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt companion/config.py companion/providers.py tests/test_providers.py
git commit -m "feat: add provider adapters for local, Claude, OpenAI, z.ai"
```

---

### Task 2: LLMClient delegates to a provider

**Files:**
- Modify: `companion/llm_client.py` (full rewrite)
- Test: `tests/test_llm_client.py` (full rewrite, same coverage)

**Interfaces:**
- Consumes: any object with `chat(system: str, turns: list[dict]) -> str` (Task 1's providers qualify).
- Produces: `LLMClient(provider, system_prompt)` with unchanged methods `send(user_text) -> str`, `reset(memory="")`, `seed_assistant(text)`, `summarize(instruction) -> str`, `has_user_turns() -> bool`. Stage-direction stripping stays in `send`.

- [ ] **Step 1: Rewrite the tests**

Replace the entire contents of `tests/test_llm_client.py` with:

```python
# tests/test_llm_client.py
from companion.llm_client import LLMClient


class FakeProvider:
    """Records chat() calls; returns queued replies (default 'OK.')."""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = []

    def chat(self, system, turns):
        self.calls.append((system, [dict(turn) for turn in turns]))
        return self.replies.pop(0) if self.replies else "OK."


def test_send_returns_reply_and_calls_provider_with_system_and_history():
    provider = FakeProvider(["Hi there!"])
    client = LLMClient(provider, "You are a helpful tutor.")

    reply = client.send("Hello")

    assert reply == "Hi there!"
    assert provider.calls == [
        ("You are a helpful tutor.", [{"role": "user", "content": "Hello"}])
    ]


def test_send_accumulates_conversation_history():
    provider = FakeProvider(["First.", "Second."])
    client = LLMClient(provider, "system prompt")

    client.send("first message")
    client.send("second message")

    _, second_turns = provider.calls[1]
    assert second_turns == [
        {"role": "user", "content": "first message"},
        {"role": "assistant", "content": "First."},
        {"role": "user", "content": "second message"},
    ]


def test_reset_clears_turns_and_restores_plain_system_prompt():
    provider = FakeProvider()
    client = LLMClient(provider, "system prompt")
    client.send("first message")

    client.reset()

    assert client.turns == []
    assert client.system == "system prompt"


def test_reset_with_memory_injects_remembered_context():
    provider = FakeProvider()
    client = LLMClient(provider, "Base prompt.")

    client.reset("- Caio visited Japan.")

    assert client.system == (
        "Base prompt.\n\nWhat you remember about the user from "
        "previous sessions:\n- Caio visited Japan."
    )
    assert client.turns == []


def test_seed_assistant_records_greeting_without_calling_provider():
    provider = FakeProvider()
    client = LLMClient(provider, "system prompt")

    client.seed_assistant("Hey Caio!")

    assert client.turns == [{"role": "assistant", "content": "Hey Caio!"}]
    assert provider.calls == []


def test_summarize_appends_instruction_without_mutating_history():
    provider = FakeProvider(["- Bullets."])
    client = LLMClient(provider, "system prompt")
    client.seed_assistant("Hi!")
    turns_before = [dict(turn) for turn in client.turns]

    result = client.summarize("Summarize the session.")

    assert result == "- Bullets."
    assert client.turns == turns_before
    _, sent_turns = provider.calls[0]
    assert sent_turns == turns_before + [
        {"role": "user", "content": "Summarize the session."}
    ]


def test_has_user_turns_false_for_fresh_session_true_after_send():
    provider = FakeProvider()
    client = LLMClient(provider, "system prompt")
    client.seed_assistant("Hi!")
    assert client.has_user_turns() is False

    client.send("Hello")
    assert client.has_user_turns() is True


def test_send_strips_stage_directions_before_storing_and_returning():
    provider = FakeProvider(["(laughs) Well, Rome sounds *smiling* amazing (pauses) ."])
    client = LLMClient(provider, "system prompt")

    reply = client.send("I visited Rome!")

    assert reply == "Well, Rome sounds amazing."
    assert client.turns[-1] == {"role": "assistant", "content": "Well, Rome sounds amazing."}


def test_send_leaves_clean_replies_untouched():
    provider = FakeProvider(["Nice! How was Rome?"])
    client = LLMClient(provider, "system prompt")

    assert client.send("I visited Rome!") == "Nice! How was Rome?"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_llm_client.py -q`
Expected: 9 failures (`TypeError` — old constructor takes a model string, no `turns`/`system` attributes).

- [ ] **Step 3: Rewrite the implementation**

Replace the entire contents of `companion/llm_client.py` with:

```python
# companion/llm_client.py
import re

# LLMs write roleplay stage directions — "(laughs)", "*smiles*" — even when
# the system prompt forbids them, and the TTS would read them aloud verbatim.
# Strip them before the reply is stored or spoken; keeping them out of the
# history also stops the model imitating its own habit.
_STAGE_DIRECTIONS = re.compile(r"\([^)]*\)|\*[^*]*\*")


def _strip_stage_directions(text: str) -> str:
    cleaned = _STAGE_DIRECTIONS.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    return cleaned.strip()


class LLMClient:
    def __init__(self, provider, system_prompt: str):
        self.provider = provider
        self.system_prompt = system_prompt
        self.system = system_prompt
        self.turns = []

    def send(self, user_text: str) -> str:
        self.turns.append({"role": "user", "content": user_text})
        # list(...) snapshots the turns at call time — providers (and
        # mock-based tests) must not observe the append below.
        reply = _strip_stage_directions(
            self.provider.chat(self.system, list(self.turns))
        )
        self.turns.append({"role": "assistant", "content": reply})
        return reply

    def reset(self, memory: str = "") -> None:
        self.system = self.system_prompt
        if memory:
            self.system += (
                "\n\nWhat you remember about the user from previous "
                "sessions:\n" + memory
            )
        self.turns = []

    def seed_assistant(self, text: str) -> None:
        self.turns.append({"role": "assistant", "content": text})

    def summarize(self, instruction: str) -> str:
        # The session is over: the instruction and reply deliberately stay
        # out of self.turns — this is a side-channel request.
        turns = list(self.turns) + [{"role": "user", "content": instruction}]
        return self.provider.chat(self.system, turns)

    def has_user_turns(self) -> bool:
        return any(turn["role"] == "user" for turn in self.turns)
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: 37 passed. (`main.py` still constructs `LLMClient(config.OLLAMA_MODEL, ...)` — that's dead-wrong now but only exercised at runtime; Task 3 fixes it. `python -c "import companion.main"` still succeeds.)

- [ ] **Step 5: Commit**

```bash
git add companion/llm_client.py tests/test_llm_client.py
git commit -m "refactor: LLMClient delegates chat to an injected provider"
```

---

### Task 3: Main wiring — menu, --brain, .env, guarded send, docs

**Files:**
- Modify: `companion/main.py`
- Modify: `.gitignore` (add `.env`)
- Modify: `README.md` (Cloud brains section)
- Test: `tests/test_main.py` (new file, `choose_brain` only)

**Interfaces:**
- Consumes: `make_provider(name)`, `REQUIRED_ENV` from Task 1; `LLMClient(provider, system_prompt)` from Task 2.
- Produces: `choose_brain(cli_choice: str | None) -> str`, `PROVIDER_NAMES = ["local", "claude", "openai", "zai"]`, `check_api_key_available(brain)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_main.py`:

```python
# tests/test_main.py
from unittest.mock import patch

from companion.main import choose_brain


def test_cli_flag_skips_the_menu():
    with patch("builtins.input") as mock_input:
        assert choose_brain("claude") == "claude"
    mock_input.assert_not_called()


def test_empty_input_returns_config_default():
    with patch("builtins.input", return_value=""):
        with patch("companion.main.config.LLM_PROVIDER", "local"):
            assert choose_brain(None) == "local"


def test_number_input_picks_from_menu():
    with patch("builtins.input", return_value="2"):
        assert choose_brain(None) == "claude"


def test_name_input_picks_provider():
    with patch("builtins.input", return_value="zai"):
        assert choose_brain(None) == "zai"


def test_garbage_input_falls_back_to_default():
    with patch("builtins.input", return_value="skynet"):
        with patch("companion.main.config.LLM_PROVIDER", "local"):
            assert choose_brain(None) == "local"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_main.py -q`
Expected: `ImportError: cannot import name 'choose_brain'`.

- [ ] **Step 3: Wire up main.py**

In `companion/main.py`:

a. Replace the import block at the top with:

```python
# companion/main.py
import argparse
import os
import sys

import numpy as np
import ollama
import sounddevice as sd
from dotenv import load_dotenv

from companion import config
from companion.llm_client import LLMClient
from companion.memory import Memory
from companion.providers import REQUIRED_ENV, make_provider
from companion.speaker import Speaker
from companion.state_machine import Action, StateMachine
from companion.transcriber import Transcriber
from companion.voice_detector import VoiceDetector

PROVIDER_NAMES = ["local", "claude", "openai", "zai"]
```

b. Add below `check_ollama_reachable`:

```python
def check_api_key_available(brain: str) -> None:
    env_var = REQUIRED_ENV[brain]
    if not os.environ.get(env_var):
        print(
            f"ERROR: {env_var} is not set. Put it in a .env file at the "
            'project root (see the README section "Cloud brains") and try again.'
        )
        sys.exit(1)


def choose_brain(cli_choice) -> str:
    if cli_choice:
        return cli_choice
    print("Choose a brain:")
    for number, name in enumerate(PROVIDER_NAMES, start=1):
        marker = " (default)" if name == config.LLM_PROVIDER else ""
        print(f"  {number}. {name}{marker}")
    answer = input("Number or name [Enter = default]: ").strip().lower()
    if not answer:
        return config.LLM_PROVIDER
    if answer.isdigit() and 1 <= int(answer) <= len(PROVIDER_NAMES):
        return PROVIDER_NAMES[int(answer) - 1]
    if answer in PROVIDER_NAMES:
        return answer
    print(f"Unknown choice '{answer}', using {config.LLM_PROVIDER}.")
    return config.LLM_PROVIDER
```

c. Replace the start of `main()` — everything from `def main() -> None:` through the `check_voice...`/`check_tts_files_available()` call — with:

```python
def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Voice English companion")
    parser.add_argument(
        "--brain",
        choices=PROVIDER_NAMES,
        default=None,
        help="skip the startup menu and use this provider",
    )
    args = parser.parse_args()
    brain = choose_brain(args.brain)
    print(f"Brain: {brain}")

    print("Checking services and microphone...")
    if brain == "local":
        # Cloud brains never touch Ollama, so the llama model stays out of
        # VRAM — that's the point of using them while gaming.
        check_ollama_reachable()
    else:
        check_api_key_available(brain)
    check_microphone_available()
    check_tts_files_available()
```

d. Replace `llm = LLMClient(config.OLLAMA_MODEL, config.SYSTEM_PROMPT)` with:

```python
    llm = LLMClient(make_provider(brain), config.SYSTEM_PROMPT)
```

e. Replace the FORWARD branch with:

```python
            elif action == Action.FORWARD:
                try:
                    reply = llm.send(text)
                except Exception as exc:
                    # Cloud APIs hiccup and Ollama can die mid-session; keep
                    # the session alive. (The dangling user turn is harmless:
                    # every provider accepts consecutive user messages.)
                    print(f"WARNING: The brain failed to reply ({exc}).")
                    speak_safely(
                        speaker, "Sorry, I had trouble thinking. Say that again?"
                    )
                    continue
                print(f"Companion: {reply}")
                speak_safely(speaker, reply)
```

- [ ] **Step 4: Hygiene and docs**

Add `.env` on its own line to `.gitignore`.

In `README.md`, insert before the "Usage notes" section:

````markdown
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
````

- [ ] **Step 5: Verify**

Run: `python -m pytest -q` — expected: 42 passed.
Run: `python -c "import companion.main"` — expected: exit 0.
Check: `.env` appears in `.gitignore`.

- [ ] **Step 6: Commit**

```bash
git add companion/main.py tests/test_main.py .gitignore README.md
git commit -m "feat: brain menu, --brain flag, .env keys, guarded send"
```

---

## Final verification (after Task 3)

1. `python -m pytest -q` → 42 passed.
2. Smoke test each cloud provider the user has a key for (one real exchange
   through `LLMClient` + provider, no audio needed).
3. Manual acceptance (user): launch, pick a cloud brain from the menu, hold
   a conversation; relaunch with `--brain local` and confirm llama still
   works; check `memory.md` still updates on "bye bye" in both modes.
