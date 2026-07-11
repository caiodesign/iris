# tests/test_main.py
from unittest.mock import patch

from companion.main import (
    PROVIDER_NAMES,
    STT_NAMES,
    ask_yes_no,
    choose_from_menu,
    print_event,
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


def test_print_event_formats_each_kind(capsys):
    print_event({"event": "heard", "text": "hi"})
    print_event({"event": "reply", "text": "hello"})
    print_event({"event": "system", "text": "Waking up."})
    print_event({"event": "warning", "text": "oops"})
    print_event({"event": "error", "text": "bad"})
    print_event({"event": "status", "state": "listening"})  # silent
    print_event({"event": "session_ended"})  # silent
    out = capsys.readouterr().out
    assert "Heard: hi" in out
    assert "Iris: hello" in out
    assert "Waking up." in out
    assert "WARNING: oops" in out
    assert "ERROR: bad" in out
    assert "listening" not in out
