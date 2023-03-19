"""Tests for FakeLLM."""
import pytest

from langchain_contrib.llms import FakeLLM


def test_fake_foo() -> None:
    """Test that by default with no configuration, the FakeLLM returns 'foo'."""
    llm = FakeLLM()
    assert llm("Dummy prompt") == "foo"


def test_fake_mapping() -> None:
    """Test that FakeLLM can return cached responses when specified."""
    llm = FakeLLM(mapped_responses={"Hello": "world"})
    assert llm("Hello") == "world"
    assert llm("Some other prompt") == "foo"


def test_fake_sequence() -> None:
    """Test that FakeLLM can return sequenced responses when specified."""
    llm = FakeLLM(sequenced_responses=["One", "Two", "Three"])
    assert llm("Count") == "One"
    assert llm("Count") == "Two"
    assert llm("Count") == "Three"
    assert llm("Count") == "foo"


def test_no_stop() -> None:
    """Test that a lack of stops will raise an error."""
    llm = FakeLLM(check_stops=True)
    with pytest.raises(AssertionError):
        llm("Can't stop won't stop")


def test_wrong_stop() -> None:
    """Test that the wrong stop will raise an error."""
    llm = FakeLLM(check_stops=True, sequenced_responses=["No stop"])
    with pytest.raises(AssertionError):
        llm("Can't stop won't stop", stop=["."])


def test_right_stop() -> None:
    """Test that the right stop will not raise an error."""
    llm = FakeLLM(check_stops=True, sequenced_responses=["Stop removed."])
    assert llm("Can't stop won't stop", stop=["."]) == "Stop removed"
