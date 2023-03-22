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


def test_make_choices() -> None:
    """Test that the choice chain can pick different choices with different outputs."""

    def choice(inputs: Dict[str, str]) -> Dict[str, str]:
        return {"choice": inputs["c"]}

    choices = ChoiceChain(
        choice_picker=FakeChain(
            expected_inputs=["c"],
            expected_outputs=["choice"],
            inputs_to_outputs=choice,
        ),
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

    def choice(inputs: Dict[str, str]) -> Dict[str, str]:
        return {"choice": inputs["c"]}

    def prep_picker(picker_outputs: Dict[str, str]) -> Dict[str, str]:
        if picker_outputs["choice"] == "first":
            return {"a_input": "foo"}
        else:
            return {"b_input": "bar"}

    choices = ChoiceChain(
        choice_picker=FakeChain(output={"choice": "second", "extra": "argument"}),
        prep_picker_output=prep_picker,
        choices={
            "first": FakeChain(output={"a": "one"}),
            "second": FakeChain(output={"b": "two"}),
        },
    )
    choices = ChoiceChain(
        choice_picker=FakeChain(
            expected_inputs=["c"],
            expected_outputs=["choice"],
            inputs_to_outputs=choice,
        ),
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
