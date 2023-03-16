"""Patch for Terminal class."""

from typing import Callable

from vcr.cassette import Cassette
from vcr_langchain.patch import GenericPatch, add_patchers

from ..terminal import Terminal


class TerminalPatch(GenericPatch):
    """Patch for Terminal class."""

    def __init__(self, cassette: Cassette) -> None:
        """Initialize patch."""
        super().__init__(cassette, Terminal, "_get_raw_shell_update")

    def get_same_signature_override(self) -> Callable:
        """Obtain same-signature override for Terminal._get_raw_shell_update."""

        def _call(og_self: Terminal, cmd: str) -> str:
            return self.generic_override(og_self, cmd=cmd)

        return _call


add_patchers(TerminalPatch)
