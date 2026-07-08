# tests/test_main.py
from unittest.mock import patch

from companion.main import choose_brain


def test_cli_flag_skips_the_menu():
    with patch("builtins.input") as mock_input:
        assert choose_brain("claude") == "claude"
    mock_input.assert_not_called()


def test_empty_input_returns_config_default():
    with patch("builtins.input", return_value=""):
        with patch("companion.main.config.LLM_PROVIDER", "local"):
            assert choose_brain(None) == "local"


def test_number_input_picks_from_menu():
    with patch("builtins.input", return_value="2"):
        assert choose_brain(None) == "claude"


def test_name_input_picks_provider():
    with patch("builtins.input", return_value="zai"):
        assert choose_brain(None) == "zai"


def test_garbage_input_falls_back_to_default():
    with patch("builtins.input", return_value="skynet"):
        with patch("companion.main.config.LLM_PROVIDER", "local"):
            assert choose_brain(None) == "local"
