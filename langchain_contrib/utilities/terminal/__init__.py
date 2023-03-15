"""Terminal with persistent shell between commands."""

from .terminal import Terminal

try:
    from vcr_langchain.patch import add_patchers

    from .patcher import TerminalPatch

    add_patchers(TerminalPatch)
except ImportError:
    pass  # vcr_langchain is an optional dependency

__all__ = [
    "Terminal",
]
