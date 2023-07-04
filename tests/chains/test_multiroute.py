"""Test that ZMultiRouteChain can successfully make choices."""


from typing import Any, List, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.tools.python.tool import PythonREPLTool
from pydantic import BaseModel

from langchain_contrib.chains import ZMultiRouteChain
from langchain_contrib.chains.testing import FakeChain, FakeRouterChain
from langchain_contrib.tools import TerminalTool


def test_make_choices() -> None:
    """Test that ZMultiRouteChain can pick different choices with different outputs."""
    choices = ZMultiRouteChain(
        router_chain=FakeRouterChain(expected_inputs=["destination"]),
        default_chain=FakeChain(),
        destination_chains={
            "first": FakeChain(output={"a": "one"}),
            "second": FakeChain(output={"b": "two"}),
        },
    )
    assert choices({"destination": "first"}) == {
        "destination": "first",
        "next_inputs": {},
        "a": "one",
    }
    assert choices({"destination": "second"}) == {
        "destination": "second",
        "next_inputs": {},
        "b": "two",
    }


def test_separated_io() -> None:
    """Test that ZMultiRouteChain can retain input/output information.

    Input information should be retained even when output contains the same keys.
    """
    choices = ZMultiRouteChain(
        router_chain=FakeRouterChain(),
        default_chain=FakeChain(),
        destination_chains={
            "first": FakeChain(output={"a": "one"}),
            "second": FakeChain(output={"b": "two"}),
        },
    )

    assert choices({"destination": "first", "next_inputs": {"a": 1}}) == {
        "destination": "first",
        "next_inputs": {"a": 1},
        "a": "one",
    }

    assert choices({"destination": "second", "next_inputs": {"a": 2}}) == {
        "destination": "second",
        "next_inputs": {"a": 2},
        "a": 2,
        "b": "two",
    }


def test_tools() -> None:
    """Test that ZMultiRouteChain can be loaded and run from tools."""
    chain = ZMultiRouteChain.from_tools(
        FakeRouterChain(),
        [PythonREPLTool(), TerminalTool()],  # type: ignore
        verbose=True,
    )
    assert chain({"destination": "Python_REPL", "next_inputs": "print(2 ** 16)"}) == {
        "destination": "Python_REPL",
        "next_inputs": "print(2 ** 16)",
        "action_input": "print(2 ** 16)",
        "action_result": "65536\n",
    }


class ColorCheckingHandler(BaseCallbackHandler, BaseModel):
    """A custom callback handler for testing colors in tool logging."""

    expected_colors: List[str]
    tool_end_count: int = 0
    raise_error: bool = True

    def on_tool_end(
        self, output: str, *, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Check that the right color is being set."""
        self.tool_end_count += 1
        assert color == self.expected_colors.pop(0)


def test_tool_colors() -> None:
    """Test that ZMultiRouteChain sets tool colors properly."""
    chain = ZMultiRouteChain.from_tools(
        FakeRouterChain(),
        [PythonREPLTool(), TerminalTool()],  # type: ignore
    )
    callback = ColorCheckingHandler(expected_colors=["blue", "yellow"])
    chain(
        {"destination": "Python_REPL", "next_inputs": "print(2 ** 16)"},
        callbacks=[callback],
    )
    chain(
        {"destination": "Terminal", "next_inputs": "echo 'hello world'"},
        callbacks=[callback],
    )
    assert callback.tool_end_count == 2
