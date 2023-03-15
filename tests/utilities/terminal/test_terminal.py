"""Test the Terminal class."""

from langchain_contrib.utilities.terminal import Terminal
from langchain_contrib.utils.tests import current_directory


def test_terminal_simple_bash() -> None:
    """Test a simple bash command."""
    t = Terminal()
    assert t.run_bash_command("ls Makefile") == "Makefile\n"


def test_directory_change() -> None:
    """Test changing the directory."""
    with current_directory():  # reset to present cwd after test
        t = Terminal()
        assert t.run_bash_command("pwd").strip().endswith("langchain-contrib")
        t.run_bash_command("cd tests")
        assert t.run_bash_command("pwd").strip().endswith("langchain-contrib/tests")
