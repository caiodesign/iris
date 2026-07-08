# tests/test_main.py
from unittest.mock import patch

from companion.main import PROVIDER_NAMES, STT_NAMES, choose_from_menu


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
