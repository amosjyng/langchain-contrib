"""Fake chains for testing purposes."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.router import RouterChain
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
    inputs_to_outputs: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    """Function to transform inputs to outputs."""

    @property
    def input_keys(self) -> List[str]:
        """Input keys this chain expects."""
        return self.expected_inputs

    @property
    def output_keys(self) -> List[str]:
        """The keys of the predefined output dict."""
        return list(set(self.expected_outputs).union(self.output.keys()))

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Return the output dict, along with inputs if so specified."""
        if self.inputs_to_outputs is None:
            return self.output
        return self.inputs_to_outputs(inputs)


def FakePicker(input_key: str = "c", output_key: str = "choice") -> FakeChain:
    """Create a fake chain that converts the input key to the output key.

    This is a convenience function that allows you to specify arbitrary fake outputs
    from arbitrary fake inputs without having to create a new FakeChain every time.
    Especially useful for testing `ChoiceChain`s.
    """
    return FakeChain(
        expected_inputs=[input_key],
        expected_outputs=[output_key],
        inputs_to_outputs=lambda inputs: {output_key: inputs[input_key]},
    )


def fake_router_inputs_to_outputs_fn(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Pass-through function to allow tester to specify exact inputs."""
    if "destination" not in inputs:
        inputs["destination"] = None
    if "next_inputs" not in inputs:
        inputs["next_inputs"] = {}
    return inputs


class FakeRouterChain(RouterChain, FakeChain):
    """Fake router chain that returns predefined outputs."""

    expected_inputs: List[str] = ["destination", "next_inputs"]
    inputs_to_outputs: Callable[
        [Dict[str, Any]], Dict[str, Any]
    ] = fake_router_inputs_to_outputs_fn
    """Pass-through function to allow tester to specify exact inputs."""
