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
