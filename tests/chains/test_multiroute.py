"""Test that ZMultiRouteChain can successfully make choices."""


from langchain.tools.python.tool import PythonREPLTool

from langchain_contrib.chains import ToolChain, ZMultiRouteChain
from langchain_contrib.chains.testing import FakeChain, FakeRouterChain
from langchain_contrib.tools import TerminalTool, ZBaseTool


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


def test_tool_colors() -> None:
    """Test that ZMultiRouteChain sets tool colors properly."""
    chain = ZMultiRouteChain.from_tools(
        FakeRouterChain(),
        [PythonREPLTool(), TerminalTool()],  # type: ignore
    )
    colors = set()
    for choice in chain.destination_chains.values():
        assert isinstance(choice, ToolChain)
        tool = choice.tool
        assert isinstance(tool, ZBaseTool)
        colors.add(tool.color)
    assert len(colors) == 2
