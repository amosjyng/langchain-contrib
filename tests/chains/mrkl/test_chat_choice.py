"""Test MRKL agent decision execution for chat prompts."""
from typing import Dict

import pytest
from langchain.chat_models.openai import ChatOpenAI

from langchain_contrib.chains.mrkl import MrklLoopChain
from langchain_contrib.llms.testing import FakeLLM
from langchain_contrib.tools import load_tools
from langchain_contrib.utils import current_directory

vcr = pytest.importorskip("vcr_langchain")


@vcr.use_cassette()
async def test_chat_choice_live() -> Dict[str, str]:
    """Check that the MRKL chain can decide and execute an action."""
    with current_directory():
        llm = ChatOpenAI()  # type: ignore
        tools = load_tools(["persistent_terminal"])
        loop = MrklLoopChain.from_llm_and_tools(
            llm=llm,
            tools=tools,
            embed_scratchpad=True,
        )
        result = loop(
            {
                "input": (
                    "List the folders in the current directory. Enter into one of "
                    "them. List folders again."
                ),
                "agent_scratchpad": "",
            },
            return_only_outputs=True,
        )
        assert result["action_result"] == "dist/  docs/  langchain_contrib/  tests/\n"
        return result


@vcr.use_cassette()
def test_faked_tool_usage() -> None:
    """Check that the MRKL chain can use a tool.

    Uses fake input.
    """
    llm = FakeLLM(
        sequenced_responses=[
            """
I need to run date.
Action: Terminal
Action Input: date
""".lstrip()
        ]
    )
    tools = load_tools(["persistent_terminal"])
    loop = MrklLoopChain.from_llm_and_tools(
        llm=llm,
        tools=tools,
        embed_scratchpad=True,
    )
    result = loop(
        {
            "input": "What is the time?",
            "agent_scratchpad": "",
        },
        return_only_outputs=True,
    )
    assert result["choice"] == "Terminal"
    assert result["thought"] == "I need to run date."
    assert result.choice_inputs == {"action_input": "date"}
    assert result.choice_outputs == {"action_result": "Thu Apr  6 20:30:55 AEST 2023\n"}


def test_faked_final_action() -> None:
    """Check that the MRKL chain can end the loop."""
    llm = FakeLLM(
        sequenced_responses=[
            """
I now know the time.
Final Answer: 8 PM.
""".lstrip()
        ]
    )
    tools = load_tools(["persistent_terminal"])
    loop = MrklLoopChain.from_llm_and_tools(
        llm=llm,
        tools=tools,
        embed_scratchpad=True,
    )
    result = loop(
        {
            "input": "What is the time?",
            "agent_scratchpad": "...",
        },
        return_only_outputs=True,
    )
    assert result["choice"] == "Final Answer"
    assert result["thought"] == "I now know the time."
    assert result.choice_inputs == {"action_input": "8 PM."}
    assert result.choice_outputs == {"answer": "8 PM."}


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(test_chat_choice_live)
