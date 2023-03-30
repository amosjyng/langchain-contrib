"""Test that the ChoiceChain can successfully make choices."""

from typing import Dict

import pytest

from langchain_contrib.chains import ChoiceChain
from langchain_contrib.chains.testing import FakeChain


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


def choice(inputs: Dict[str, str]) -> Dict[str, str]:
    """Convert the 'c' key to 'choice'."""
    return {"choice": inputs["c"]}


fake_picker = FakeChain(
    expected_inputs=["c"],
    expected_outputs=["choice"],
    inputs_to_outputs=choice,
)


def test_make_choices() -> None:
    """Test that the choice chain can pick different choices with different outputs."""
    choices = ChoiceChain(
        choice_picker=fake_picker,
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
        choice_picker=fake_picker,
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
        choice_picker=fake_picker,
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
