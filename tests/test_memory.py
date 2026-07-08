# tests/test_memory.py
from companion.memory import Memory


def test_load_returns_empty_string_when_file_missing(tmp_path):
    memory = Memory(str(tmp_path / "memory.md"), 6000)
    assert memory.load() == ""


def test_append_session_writes_dated_heading_and_summary(tmp_path):
    path = tmp_path / "memory.md"
    memory = Memory(str(path), 6000)

    memory.append_session("- Talked about food.\n- Caio visited Japan.")

    content = path.read_text(encoding="utf-8")
    assert content.startswith("## 20")  # "## 2026-07-08 14:30" style heading
    assert "- Talked about food.\n- Caio visited Japan." in content


def test_append_session_accumulates_sessions(tmp_path):
    path = tmp_path / "memory.md"
    memory = Memory(str(path), 6000)

    memory.append_session("- First session.")
    memory.append_session("- Second session.")

    content = path.read_text(encoding="utf-8")
    assert "- First session." in content
    assert "- Second session." in content
    assert content.count("## 20") == 2


def test_load_returns_at_most_max_chars_keeping_the_end(tmp_path):
    path = tmp_path / "memory.md"
    memory = Memory(str(path), 20)

    memory.append_session("A" * 30)
    loaded = memory.load()

    assert len(loaded) <= 20
    assert loaded == "A" * 20  # the END of the file survives truncation
