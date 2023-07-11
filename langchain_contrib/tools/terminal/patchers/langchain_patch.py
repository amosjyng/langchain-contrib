"""Patch for langchain tooling."""

from langchain.agents.load_tools import _BASE_TOOLS

from langchain_contrib.tools.terminal.tool import TerminalTool


def _get_smart_terminal() -> TerminalTool:
    return TerminalTool()


_BASE_TOOLS["smart_terminal"] = _get_smart_terminal
