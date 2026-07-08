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
