# tests/test_memory.py
import os

from companion.memory import Memory


def _memory(tmp_path, timeline_max_chars=4000):
    return Memory(str(tmp_path / "memory"), timeline_max_chars)


def test_load_returns_empty_string_when_nothing_stored(tmp_path):
    assert _memory(tmp_path).load() == ""


def test_load_durable_returns_empty_string_when_file_missing(tmp_path):
    assert _memory(tmp_path).load_durable() == ""


def test_append_timeline_writes_dated_heading_and_entry(tmp_path):
    memory = _memory(tmp_path)

    memory.append_timeline("- Talked about food.")

    content = (tmp_path / "memory" / "timeline.md").read_text(encoding="utf-8")
    assert content.startswith("## 20")  # "## 2026-07-08 14:30" style heading
    assert "- Talked about food." in content


def test_append_timeline_accumulates_sessions(tmp_path):
    memory = _memory(tmp_path)

    memory.append_timeline("- First session.")
    memory.append_timeline("- Second session.")

    content = (tmp_path / "memory" / "timeline.md").read_text(encoding="utf-8")
    assert "- First session." in content
    assert "- Second session." in content
    assert content.count("## 20") == 2


def test_write_durable_persists_and_load_durable_reads_it_back(tmp_path):
    memory = _memory(tmp_path)

    memory.write_durable("## Facts\n- Caio likes ramen.")

    assert memory.load_durable() == "## Facts\n- Caio likes ramen."


def test_write_durable_overwrites_previous_content(tmp_path):
    memory = _memory(tmp_path)

    memory.write_durable("## Facts\n- Old fact.")
    memory.write_durable("## Facts\n- New fact.")

    assert memory.load_durable() == "## Facts\n- New fact."


def test_write_durable_ignores_empty_or_whitespace_input(tmp_path):
    memory = _memory(tmp_path)

    memory.write_durable("## Facts\n- Keep me.")
    memory.write_durable("   \n  ")  # must NOT blank out accumulated facts

    assert memory.load_durable() == "## Facts\n- Keep me."


def test_write_durable_leaves_no_temp_file_behind(tmp_path):
    memory = _memory(tmp_path)

    memory.write_durable("## Facts\n- Caio likes ramen.")

    files = os.listdir(tmp_path / "memory")
    assert files == ["durable.md"]  # atomic rename left no .tmp artifact


def test_load_returns_durable_only_when_no_timeline(tmp_path):
    memory = _memory(tmp_path)

    memory.write_durable("## Facts\n- Caio likes ramen.")

    assert memory.load() == "## Facts\n- Caio likes ramen."


def test_load_assembles_durable_then_recent_sessions(tmp_path):
    memory = _memory(tmp_path)

    memory.write_durable("## Facts\n- Caio likes ramen.")
    memory.append_timeline("- Talked about food.")

    loaded = memory.load()
    assert loaded == (
        "## Facts\n- Caio likes ramen.\n\n"
        "Recent sessions:\n"
        + (tmp_path / "memory" / "timeline.md").read_text(encoding="utf-8").strip()
    )
    # Durable comes first, timeline second under a "Recent sessions:" header.
    assert loaded.index("## Facts") < loaded.index("Recent sessions:")


def test_load_truncates_timeline_to_the_tail_keeping_the_end(tmp_path):
    memory = Memory(str(tmp_path / "memory"), timeline_max_chars=20)

    memory.append_timeline("B" * 40)
    loaded = memory.load()

    # Only the timeline is windowed; it keeps the END of the file.
    assert "Recent sessions:\n" in loaded
    tail = loaded.split("Recent sessions:\n", 1)[1]
    assert len(tail) <= 20
    assert tail == "B" * 20
