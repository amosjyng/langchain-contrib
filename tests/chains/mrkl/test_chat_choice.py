"""Test MRKL agent decision execution for chat prompts."""

from typing import Dict

import pytest
from langchain.chat_models.openai import ChatOpenAI

from langchain_contrib.chains.mrkl import MrklLoopChain
from langchain_contrib.tools import load_tools
from langchain_contrib.utils import current_directory

vcr = pytest.importorskip("vcr_langchain")


@vcr.use_cassette()
async def test_chat_choice() -> Dict[str, str]:
    """Check that the MRKL chain can decide and execute an action."""
    with current_directory():
        llm = ChatOpenAI()  # type: ignore
        tools = load_tools(["persistent_terminal"])
        loop = MrklLoopChain.from_llm_and_tools(llm=llm, tools=tools)
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
        assert result["action_result"] == "dist/  docs/  langchain_contrib/  tests/"
        return result


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(test_chat_choice)
