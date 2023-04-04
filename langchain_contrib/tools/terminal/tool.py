"""Make Terminal available in langchain Tool form."""

from pydantic import Field

from langchain_contrib.tools.z_base import ZBaseTool

from .terminal import Terminal


class TerminalTool(ZBaseTool):
    """Access to the Terminal in langchain's default Tool form."""

    name: str = "Terminal"
    description: str = (
        "Executes commands in a terminal. Input should be valid commands, and the "
        "output will be any output from running that command."
    )
    terminal: Terminal = Field(default_factory=Terminal)

    def _run(self, tool_input: str) -> str:
        """Use the terminal."""
        return self.terminal.run_bash_command(tool_input)

    async def _arun(self, tool_input: str) -> str:
        """Use the terminal asynchronously."""
        return self._run(tool_input)
