# companion/llm_client.py
import ollama


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
        reply = response["message"]["content"]
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
