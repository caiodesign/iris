# companion/llm_client.py
import re

import ollama

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


class LLMClient:
    def __init__(self, model: str, system_prompt: str):
        self.model = model
        self.system_prompt = system_prompt
        self.history = [{"role": "system", "content": system_prompt}]

    def send(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        # list(...) snapshots history at call time — the live list is mutated
        # by the append below, and callers (and mock-based tests) must not
        # observe that mutation. Do not "simplify" back to messages=self.history.
        response = ollama.chat(model=self.model, messages=list(self.history))
        reply = _strip_stage_directions(response["message"]["content"])
        self.history.append({"role": "assistant", "content": reply})
        return reply

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

    def seed_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})
