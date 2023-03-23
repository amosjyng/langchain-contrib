"""Chain that chooses and performs the next action."""
from abc import ABC
from typing import Callable, Dict, List, Mapping

from langchain.chains.base import Chain
from pydantic import BaseModel

from langchain_contrib.utils import safe_inputs


class ChoiceChain(Chain, BaseModel, ABC):
    """Chain that asks the LLM for a decision and executes it."""

    choice_picker: Chain
    """The chain that actually prompts the LLM for the choice."""
    prep_picker_output: Callable[[Dict[str, str]], Dict[str, str]] = lambda x: x
    """Interprets output from the picker chain for the chosen chain.

    Override this to do additional dict munging before it gets passed through to
    the chosen chain.
    """
    choices: Mapping[str, Chain]
    """The chains that will be run depending on the LLM's choice.

    This is a mapping from which LLM output corresponds to which chain.
    """
    choice_key: str = "choice"
    """choice_picker output key that tells us which choice was picked."""
    ignore_keys: List[str] = []
    """Keys that will be returned in final output, but not passed on to chosen chain."""

    @property
    def input_keys(self) -> List[str]:
        """Input keys to this chain."""
        return self.choice_picker.input_keys

    @property
    def output_keys(self) -> List[str]:
        """Possible output keys produced by this chain."""
        all_keys = set(self.choice_picker.output_keys)
        for choice in self.choices.values():
            all_keys.update(choice.output_keys)
        return list(all_keys)

    def _validate_outputs(self, outputs: Dict[str, str]) -> None:
        """Skip validation because different options may produce different outputs.

        It is assumed that each individual choice chain will validate its own output.
        """

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Run the logic of this chain and return the output."""
        raw_picker_output = self.choice_picker(inputs, return_only_outputs=True)
        ignored_output = {
            k: v for k, v in raw_picker_output.items() if k in self.ignore_keys
        }
        real_raw_output = {
            k: v for k, v in raw_picker_output.items() if k not in self.ignore_keys
        }

        picker_output = self.prep_picker_output(real_raw_output)
        if self.choice_key not in picker_output:
            raise KeyError(f"Choice-picking chain did not emit '{self.choice_key}'")
        choice = picker_output.pop(self.choice_key)
        assert isinstance(choice, str), f"Choice '{choice}' is not a str"
        if choice not in self.choices:
            raise KeyError(f"Choice picked does not exist: '{choice}'")
        chosen_chain = self.choices[choice]
        unused_args = picker_output.keys() - chosen_chain.input_keys
        if unused_args:
            raise KeyError(f"Extra input keys for choice: {unused_args}")

        full_inputs = {**inputs, **picker_output}
        chain_inputs = safe_inputs(chosen_chain, full_inputs)
        chain_output = chosen_chain(chain_inputs)
        return {
            self.choice_key: choice,
            **ignored_output,
            **picker_output,
            **chain_output,
        }
