"""Test the Terminal class."""

from langchain_contrib.tools.terminal import Terminal
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


def test_tabbed_script() -> None:
    """Check that escaped tabbed output is captured."""
    t = Terminal()
    assert t.run_bash_command("tests/resources/tabbed.sh") == "\\ta\n"


def test_tabbed_file() -> None:
    """Check that unescaped tab output is captured."""
    t = Terminal()
    assert t.run_bash_command("cat tests/resources/tabbed.txt") == "\ta\n"
