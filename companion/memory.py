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
