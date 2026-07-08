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
