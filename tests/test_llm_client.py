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
