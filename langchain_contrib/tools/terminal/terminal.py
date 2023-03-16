"""Module to interact with a virtual terminal."""

import os
import shlex
import time
from typing import Any

import pexpect
from pydantic import BaseModel

from .ansi_escapes import remove_ansi_escapes


class UnknownResult(Exception):
    """Exception raised when terminal output is not as expected."""


class Terminal(BaseModel):
    """A virtual terminal that supports interactive shell commands.

    This will actually change the current program's working directory in response to
    `cd` commands. This allows the LLM to subsequently edit files according to their
    relative paths in the new directory.
    """

    shell: pexpect.spawn
    """The actual shell we're interacting with."""
    bash_prompt: str
    """The prompt of the shell that tells us the last command has finished running."""
    refresh_interval: float
    """How frequently we should check for shell updates."""
    output_size: int
    """The maximum number of shell output characters to read at once."""

    class Config:
        """pydantic config object."""

        arbitrary_types_allowed = True

    def __init__(
        self,
        bash_prompt: str = "zamm$ ",
        refresh_interval: float = 0.1,
        init_delay: float = 0.1,
        output_size: int = 1000,
        **kwargs: Any,
    ) -> None:
        """Initialize a virtual terminal.

        Args:
            refresh_interval: How long to wait between terminal reads.
            init_delay: How long to wait for initial terminal prompt during init.
            output_size: How many characters to read at a time.
            bash_prompt: Constant Bash prompt to use for terminal.
            **kwargs: Additional arguments to pass to `BaseModel`.
        """
        os.environ["PS1"] = bash_prompt
        sh = pexpect.spawn("/bin/bash --norc", encoding="utf-8")
        time.sleep(init_delay)
        bash_prompt = sh.read_nonblocking(size=output_size)
        super().__init__(
            refresh_interval=refresh_interval,
            output_size=output_size,
            shell=sh,
            bash_prompt=bash_prompt,
            **kwargs,
        )

    @property
    def prompt_length(self) -> int:
        """Get the length of the terminal prompt."""
        return len(self.bash_prompt)

    def _get_raw_shell_update_uncached(self, cmd: str) -> str:
        """Get the raw terminal output for a command."""
        self.shell.sendline(cmd)
        results = ""
        try:
            while not results.endswith(self.bash_prompt):
                latest_output = self.shell.read_nonblocking(size=self.output_size)
                results += latest_output
                time.sleep(0.1)
        except pexpect.TIMEOUT as e:
            raise UnknownResult(
                "Terminal output does not have initial prompt of: "
                f"'{self.bash_prompt}':\n\n{results}"
            ) from e

        # todo: more robust way of syncing terminal actions to action chain state
        parsed_cmd = shlex.split(cmd)
        if len(parsed_cmd) == 2 and parsed_cmd[0] == "cd":
            os.chdir(parsed_cmd[1])

        return results

    def _get_raw_shell_update(self, cmd: str) -> str:
        """Get the raw terminal output for a command.

        Call this function for terminal commands that can/should be cached.
        """
        return self._get_raw_shell_update_uncached(cmd)

    def _is_terminal_state_command(self, cmd: str) -> bool:
        """Check if a command changes the terminal state."""
        return cmd.startswith("cd ")

    def _get_shell_update(self, cmd: str) -> str:
        """Get the raw terminal output for a command.

        Results may or may not be cached, depending on whether the command is known to
        modify shell state.
        """
        if self._is_terminal_state_command(cmd):
            return self._get_raw_shell_update_uncached(cmd)
        return self._get_raw_shell_update(cmd)

    def run_bash_command(self, cmd: str) -> str:
        """Run a command in the terminal.

        Returns the interpreted output of the command as a single string.

        Args:
            cmd: The command to run.

        Raises:
            UnknownResult: If the terminal output does not end with the
                expected prompt.
        """
        results = self._get_shell_update(cmd)
        assert results.startswith(cmd), (
            f"'{results}' does not start with '{cmd}'. "
            "Is non-ASCII terminal input involved?"
        )
        unix_results = results.replace("\r\n", "\n")
        # + 1 to remove leading "\n" after command input
        output = unix_results[len(cmd) + 1 : -self.prompt_length]
        without_ansi = remove_ansi_escapes(output)
        return without_ansi
