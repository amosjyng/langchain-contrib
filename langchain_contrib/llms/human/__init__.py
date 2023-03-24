"""Module for the human LLM."""

from .llm import BaseHuman, Human

try:
    from . import patchers  # noqa: F401
except ImportError:
    pass  # vcr_langchain is an optional dependency

__all__ = [
    "BaseHuman",
    "Human",
]
