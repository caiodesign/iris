# tests/test_main.py
from unittest.mock import MagicMock, patch

from companion.main import (
    PROVIDER_NAMES,
    STT_NAMES,
    ask_yes_no,
    choose_from_menu,
    remember_session,
)


def test_cli_flag_skips_the_menu():
    with patch("builtins.input") as mock_input:
        assert choose_from_menu("brain", PROVIDER_NAMES, "local", "claude") == "claude"
    mock_input.assert_not_called()


def test_empty_input_returns_default():
    with patch("builtins.input", return_value=""):
        assert choose_from_menu("brain", PROVIDER_NAMES, "local", None) == "local"


def test_number_input_picks_from_menu():
    with patch("builtins.input", return_value="2"):
        assert choose_from_menu("brain", PROVIDER_NAMES, "local", None) == "claude"


def test_name_input_picks_choice():
    with patch("builtins.input", return_value="zai"):
        assert choose_from_menu("brain", PROVIDER_NAMES, "local", None) == "zai"


def test_garbage_input_falls_back_to_default():
    with patch("builtins.input", return_value="skynet"):
        assert choose_from_menu("brain", PROVIDER_NAMES, "local", None) == "local"


def test_ears_menu_number_picks_openai():
    with patch("builtins.input", return_value="2"):
        assert choose_from_menu("ears", STT_NAMES, "local", None) == "openai"


def test_ears_cli_flag_skips_menu():
    with patch("builtins.input") as mock_input:
        assert choose_from_menu("ears", STT_NAMES, "local", "openai") == "openai"
    mock_input.assert_not_called()


def test_remember_session_appends_timeline_and_merges_durable():
    llm = MagicMock()
    llm.has_user_turns.return_value = True
    # First summarize call -> timeline entry, second -> merged durable memory.
    llm.summarize.side_effect = ["- Talked about food.", "## Facts\n- Likes ramen."]
    memory = MagicMock()
    memory.load_durable.return_value = "## Facts\n- Old fact."

    remember_session(llm, memory)

    assert llm.summarize.call_count == 2
    memory.append_timeline.assert_called_once_with("- Talked about food.")
    memory.write_durable.assert_called_once_with("## Facts\n- Likes ramen.")


def test_remember_session_does_nothing_without_user_turns():
    llm = MagicMock()
    llm.has_user_turns.return_value = False
    memory = MagicMock()

    remember_session(llm, memory)

    llm.summarize.assert_not_called()
    memory.append_timeline.assert_not_called()
    memory.write_durable.assert_not_called()


def test_remember_session_still_merges_durable_when_timeline_call_fails():
    llm = MagicMock()
    llm.has_user_turns.return_value = True
    llm.summarize.side_effect = [Exception("network blip"), "## Facts\n- Likes ramen."]
    memory = MagicMock()
    memory.load_durable.return_value = ""

    remember_session(llm, memory)  # must not raise

    memory.append_timeline.assert_not_called()
    memory.write_durable.assert_called_once_with("## Facts\n- Likes ramen.")


def test_remember_session_keeps_timeline_when_durable_call_fails():
    llm = MagicMock()
    llm.has_user_turns.return_value = True
    llm.summarize.side_effect = ["- Talked about food.", Exception("network blip")]
    memory = MagicMock()
    memory.load_durable.return_value = ""

    remember_session(llm, memory)  # must not raise

    memory.append_timeline.assert_called_once_with("- Talked about food.")
    memory.write_durable.assert_not_called()


def test_ask_yes_no_cli_flag_skips_prompt():
    with patch("builtins.input") as mock_input:
        assert ask_yes_no("Push-to-talk?", False, True) is True
    mock_input.assert_not_called()


def test_ask_yes_no_empty_returns_default():
    with patch("builtins.input", return_value=""):
        assert ask_yes_no("Push-to-talk?", False, False) is False


def test_ask_yes_no_accepts_yes_variants():
    for value in ("y", "yes", "YES"):
        with patch("builtins.input", return_value=value):
            assert ask_yes_no("Push-to-talk?", False, False) is True


def test_ask_yes_no_treats_other_input_as_no():
    with patch("builtins.input", return_value="maybe"):
        assert ask_yes_no("Push-to-talk?", False, False) is False
