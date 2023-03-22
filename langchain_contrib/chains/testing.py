"""Fake chains for testing purposes."""

from typing import Callable, Dict, List, Optional

from langchain.chains.base import Chain
from pydantic import Extra


class FakeChain(Chain):
    """Fake chain that returns predefined outputs."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    expected_inputs: List[str] = []
    """List of input keys to expect."""
    expected_outputs: List[str] = []
    """List of output keys to expect.

    Not needed if `output` is defined.
    """
    output: Dict[str, str] = {}
    """The dict to return when this chain is called.

    This is ignored if `inputs_to_outputs` is defined.
    """
    inputs_to_outputs: Optional[Callable[[Dict[str, str]], Dict[str, str]]] = None
    """Function to transform inputs to outputs."""

    @property
    def input_keys(self) -> List[str]:
        """Input keys this chain expects."""
        return self.expected_inputs

    @property
    def output_keys(self) -> List[str]:
        """The keys of the predefined output dict."""
        return list(set(self.expected_outputs).union(self.output.keys()))

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Return the output dict, along with inputs if so specified."""
        if self.inputs_to_outputs is None:
            return self.output
        return self.inputs_to_outputs(inputs)
