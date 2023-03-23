"""Tests for FakeChain."""

from typing import Dict

from langchain_contrib.chains.testing import FakeChain


def test_return_verbatim() -> None:
    """Test that FakeChain can return predefined outputs verbatim."""
    chain = FakeChain(output={"Hello": "World"})
    assert chain({}) == {"Hello": "World"}


def test_can_take_inputs() -> None:
    """Test that FakeChain can take inputs."""
    chain = FakeChain(output={"Hello": "World"}, expected_inputs=["a"])
    assert chain({"a": "sdf"}) == {"a": "sdf", "Hello": "World"}


def test_convert_inputs() -> None:
    """Test that FakeChain can optionally transform inputs into outputs."""

    def append_exclamation(inputs: Dict[str, str]) -> Dict[str, str]:
        return {"b": inputs["a"] + "!"}

    chain = FakeChain(
        expected_inputs=["a"],
        expected_outputs=["b"],
        inputs_to_outputs=append_exclamation,
    )
    assert chain({"a": "Hello World"}, return_only_outputs=True) == {
        "b": "Hello World!"
    }
