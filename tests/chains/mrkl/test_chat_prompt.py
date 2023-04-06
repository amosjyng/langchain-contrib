"""Test MRKL agent decision making for chat prompts."""

from typing import Dict

import pytest
from langchain.chat_models.openai import ChatOpenAI

from langchain_contrib.chains.mrkl import MrklPickActionChain
from langchain_contrib.tools import load_tools
from langchain_contrib.utils import current_directory

vcr = pytest.importorskip("vcr_langchain")


@vcr.use_cassette()
async def test_chat_prompt() -> Dict[str, str]:
    """Check that the chat MRKL prompt gets the LLM to pick an action as expected."""
    with current_directory():
        llm = ChatOpenAI()  # type: ignore
        tools = load_tools(["persistent_terminal"])
        picker = MrklPickActionChain.from_llm_and_tools(
            llm=llm, tools=tools, embed_scratchpad=True
        )
        result = picker(
            {
                "input": (
                    "List the folders in the current directory. Enter into one of "
                    "them. List folders again."
                ),
                "agent_scratchpad": "",
            },
            return_only_outputs=True,
        )
        assert result["action_input"] == "ls"
        return result


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(test_chat_prompt)
