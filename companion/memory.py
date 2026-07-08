# companion/memory.py
import os
from datetime import datetime


class Memory:
    """Structured memory: a durable, LLM-managed knowledge base plus a dated,
    append-only timeline.

    - durable.md holds Facts / Goals / English sections the LLM rewrites and
      merges at session end. It is loaded in full so it never scrolls out.
    - timeline.md holds one dated entry per session, appended. Only the recent
      tail (timeline_max_chars) is loaded.
    """

    def __init__(self, dir_path: str, timeline_max_chars: int):
        self.dir_path = dir_path
        self.timeline_max_chars = timeline_max_chars
        self.durable_path = os.path.join(dir_path, "durable.md")
        self.timeline_path = os.path.join(dir_path, "timeline.md")

    def load(self) -> str:
        # Body only: llm_client.reset() supplies the "What you remember..."
        # header, so repeating it here would double up.
        durable = self.load_durable()
        timeline = self._load_timeline_tail()
        parts = []
        if durable:
            parts.append(durable)
        if timeline:
            parts.append("Recent sessions:\n" + timeline)
        return "\n\n".join(parts)

    def load_durable(self) -> str:
        if not os.path.exists(self.durable_path):
            return ""
        with open(self.durable_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def append_timeline(self, entry: str) -> None:
        os.makedirs(self.dir_path, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(self.timeline_path, "a", encoding="utf-8") as f:
            f.write(f"## {stamp}\n{entry.strip()}\n\n")

    def write_durable(self, text: str) -> None:
        # Empty-guard: a blank or failed merge must never wipe accumulated
        # facts. Atomic temp+rename: a crash mid-write can't truncate the file.
        if not text.strip():
            return
        os.makedirs(self.dir_path, exist_ok=True)
        tmp_path = self.durable_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")
        os.replace(tmp_path, self.durable_path)

    def _load_timeline_tail(self) -> str:
        if not os.path.exists(self.timeline_path):
            return ""
        with open(self.timeline_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return text[-self.timeline_max_chars :]
