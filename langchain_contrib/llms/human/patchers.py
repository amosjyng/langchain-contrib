"""Patch for Terminal class."""

from typing import Callable, List, Optional

from vcr.cassette import Cassette
from vcr_langchain.patch import GenericPatch, add_patchers

from .llm import BaseHuman, Human


class BaseHumanPatch(GenericPatch):
    """Patch for the BaseHuman class."""

    def __init__(self, cassette: Cassette) -> None:
        """Initialize patch."""
        super().__init__(cassette, BaseHuman, "_call")

    def get_same_signature_override(self) -> Callable:
        """Obtain same-signature override for BaseHuman._call."""

        def _call(
            og_self: BaseHuman, prompt: str, stop: Optional[List[str]] = None
        ) -> str:
            return self.generic_override(og_self, prompt=prompt, stop=stop)

        return _call


class HumanPatch(GenericPatch):
    """Patch for the Human class."""

    def __init__(self, cassette: Cassette) -> None:
        """Initialize patch."""
        super().__init__(cassette, Human, "_call")

    def get_same_signature_override(self) -> Callable:
        """Obtain same-signature override for Human._call."""

        def _call(og_self: Human, prompt: str, stop: Optional[List[str]] = None) -> str:
            return self.generic_override(og_self, prompt=prompt, stop=stop)

        return _call


add_patchers(BaseHumanPatch, HumanPatch)
