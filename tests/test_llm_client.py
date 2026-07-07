# tests/test_llm_client.py
from unittest.mock import patch

from companion.llm_client import LLMClient


def test_send_returns_reply_and_calls_ollama_with_history():
    fake_response = {"message": {"role": "assistant", "content": "Hi there!"}}
    with patch("companion.llm_client.ollama.chat", return_value=fake_response) as mock_chat:
        client = LLMClient("llama3.1:8b", "You are a helpful tutor.")
        reply = client.send("Hello")

    assert reply == "Hi there!"
    mock_chat.assert_called_once_with(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": "You are a helpful tutor."},
            {"role": "user", "content": "Hello"},
        ],
    )


def test_send_accumulates_conversation_history():
    fake_response = {"message": {"role": "assistant", "content": "Sure!"}}
    with patch("companion.llm_client.ollama.chat", return_value=fake_response) as mock_chat:
        client = LLMClient("llama3.1:8b", "system prompt")
        client.send("first message")
        client.send("second message")

    second_call_messages = mock_chat.call_args.kwargs["messages"]
    assert len(second_call_messages) == 4
    assert second_call_messages[-1] == {"role": "user", "content": "second message"}


def test_reset_clears_history_back_to_system_prompt():
    fake_response = {"message": {"role": "assistant", "content": "Sure!"}}
    with patch("companion.llm_client.ollama.chat", return_value=fake_response):
        client = LLMClient("llama3.1:8b", "system prompt")
        client.send("first message")
        client.reset()

    assert client.history == [{"role": "system", "content": "system prompt"}]


def test_seed_assistant_records_greeting_without_calling_ollama():
    client = LLMClient("llama3.1:8b", "system prompt")
    client.seed_assistant("Hi! What would you like to work on today?")

    assert client.history == [
        {"role": "system", "content": "system prompt"},
        {
            "role": "assistant",
            "content": "Hi! What would you like to work on today?",
        },
    ]
