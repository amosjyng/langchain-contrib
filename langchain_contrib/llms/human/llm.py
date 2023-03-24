"""Module that defines human-based 'LLM's."""

import logging
from typing import List, Optional

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from simple_term_menu import TerminalMenu

from langchain_contrib.prompts.choice import ChoiceStr


class BaseHuman(LLM):
    """Base class for humans acting as LLMs."""

    @property
    def _llm_type(self) -> str:
        return "Human"

    def _has_stop(self, input: str, stop: List[str]) -> bool:
        """Whether or not the input contains any of the current stops."""
        for s in stop:
            if s in input:
                logging.debug(f"Found '{repr(s)}' in {input}")
                return True
        return False

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Get the human to respond to the given prompt."""
        if stop is not None:
            assert isinstance(stop, list), "Stops must be provided as a list"
        user_input = input(prompt)

        if stop is None or stop == ["\n"]:
            return user_input

        while not self._has_stop(user_input, stop):
            user_input += "\n" + input()

        return enforce_stop_tokens(user_input, stop)


class Human(BaseHuman):
    """A human 'LLM' with terminal support for choice prompts."""

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Get the human to respond to the given prompt."""
        if isinstance(prompt, ChoiceStr):
            try:
                print(prompt)
                print()
                result = TerminalMenu(prompt.choices).show()
                if result is None:
                    raise ValueError("No menu response; user might've hit ctrl-C")
                return prompt.choices[result]
            except OSError:
                # likely because of: [Errno 6] No such device or address: '/dev/tty'
                return super()._call("", stop)

        return super()._call(prompt, stop)
