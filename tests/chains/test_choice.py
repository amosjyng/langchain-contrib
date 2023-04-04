"""Test that the ChoiceChain can successfully make choices."""

from typing import Dict

import pytest
from langchain.tools.python.tool import PythonREPLTool

from langchain_contrib.chains import ChoiceChain, ToolChain
from langchain_contrib.chains.testing import FakeChain, FakePicker
from langchain_contrib.tools import TerminalTool, ZBaseTool


def test_invalid_choice() -> None:
    """Test the choice chain complains if no choice key outputted."""
    choices = ChoiceChain(
        choice_picker=FakeChain(),
        choices={
            "first": FakeChain(),
            "second": FakeChain(),
        },
    )
    with pytest.raises(KeyError):
        choices({})


def test_make_choices() -> None:
    """Test that the choice chain can pick different choices with different outputs."""
    choices = ChoiceChain(
        choice_picker=FakePicker(),
        choices={
            "first": FakeChain(output={"a": "one"}),
            "second": FakeChain(output={"b": "two"}),
        },
    )
    assert choices({"c": "first"}, return_only_outputs=True) == {
        "choice": "first",
        "a": "one",
    }
    assert choices({"c": "second"}, return_only_outputs=True) == {
        "choice": "second",
        "b": "two",
    }


def test_custom_arguments() -> None:
    """Test that the choice chain can send different arguments to different choices."""

    def prep_picker(picker_outputs: Dict[str, str]) -> Dict[str, str]:
        if picker_outputs["choice"] == "first":
            return {**picker_outputs, "a_input": "foo"}
        else:
            return {**picker_outputs, "b_input": "bar"}

    choices = ChoiceChain(
        choice_picker=FakePicker(),
        prep_picker_output=prep_picker,
        choices={
            "first": FakeChain(expected_inputs=["a_input"], output={"a": "one"}),
            "second": FakeChain(expected_inputs=["b_input"], output={"b": "two"}),
        },
    )
    assert choices({"c": "first"}, return_only_outputs=True) == {
        "choice": "first",
        "a_input": "foo",
        "a": "one",
    }
    assert choices({"c": "second"}, return_only_outputs=True) == {
        "choice": "second",
        "b_input": "bar",
        "b": "two",
    }


def test_separated_io() -> None:
    """Test that the choice chain can retain input/output information."""

    def prep_picker(picker_outputs: Dict[str, str]) -> Dict[str, str]:
        if picker_outputs["choice"] == "first":
            return {**picker_outputs, "a_input": "foo"}
        else:
            return {**picker_outputs, "b_input": "bar"}

    choices = ChoiceChain(
        choice_picker=FakePicker(),
        prep_picker_output=prep_picker,
        choices={
            "first": FakeChain(expected_inputs=["a_input"], output={"a": "one"}),
            "second": FakeChain(expected_inputs=["b_input"], output={"b": "two"}),
        },
        emit_io_info=True,
    )

    first_results = choices({"c": "first"}, return_only_outputs=True)
    assert choices.chosen_inputs(first_results) == {"a_input": "foo"}
    assert choices.chosen_outputs(first_results) == {"a": "one"}

    second_results = choices({"c": "second"}, return_only_outputs=True)
    assert choices.chosen_inputs(second_results) == {"b_input": "bar"}
    assert choices.chosen_outputs(second_results) == {"b": "two"}


def test_extra_arguments() -> None:
    """Test that the choice chain will complain about extra arguments to a choice."""
    choices = ChoiceChain(
        choice_picker=FakeChain(output={"choice": "second", "extra": "argument"}),
        choices={
            "first": FakeChain(output={"a": "one"}),
            "second": FakeChain(output={"b": "two"}),
        },
    )
    with pytest.raises(KeyError):
        choices({})


def test_tools() -> None:
    """Test that the ChoiceChain can be loaded and run from tools."""
    chain = ChoiceChain.from_tools(
        FakePicker(input_key="choice"),
        [PythonREPLTool(), TerminalTool()],  # type: ignore
        verbose=True,
    )
    assert chain({"choice": "Python REPL", "action_input": "print(2 ** 16)"}) == {
        "choice": "Python REPL",
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
