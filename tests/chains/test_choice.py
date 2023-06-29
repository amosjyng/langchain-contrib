"""Test that the ChoiceChain can successfully make choices."""


from langchain.chains.router import MultiRouteChain
from langchain.tools.python.tool import PythonREPLTool

from langchain_contrib.chains import ChoiceChain, ToolChain
from langchain_contrib.chains.testing import FakeChain, FakePicker, FakeRouterChain
from langchain_contrib.tools import TerminalTool, ZBaseTool


def test_make_choices() -> None:
    """Test that the choice chain can pick different choices with different outputs."""
    choices = MultiRouteChain(
        router_chain=FakeRouterChain(),
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
    """Test that the choice chain can retain input/output information.

    Input information should be retained even when output contains the same keys.
    """
    choices = MultiRouteChain(
        router_chain=FakeRouterChain(expected_inputs=["destination", "next_inputs"]),
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
    """Test that the ChoiceChain can be loaded and run from tools."""
    chain = ChoiceChain.from_tools(
        FakePicker(input_key="choice"),
        [PythonREPLTool(), TerminalTool()],  # type: ignore
        verbose=True,
    )
    assert chain({"choice": "Python_REPL", "action_input": "print(2 ** 16)"}) == {
        "choice": "Python_REPL",
        "action_input": "print(2 ** 16)",
        "action_result": "65536\n",
    }


def test_tool_colors() -> None:
    """Test that the ChoiceChain sets tool colors properly."""
    chain = ChoiceChain.from_tools(
        FakePicker(input_key="choice"),
        [PythonREPLTool(), TerminalTool()],  # type: ignore
    )
    colors = set()
    for choice in chain.choices.values():
        assert isinstance(choice, ToolChain)
        tool = choice.tool
        assert isinstance(tool, ZBaseTool)
        colors.add(tool.color)
