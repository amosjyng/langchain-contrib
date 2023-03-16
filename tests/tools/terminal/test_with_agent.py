"""Test agent usage of terminal."""

import langchain_visualizer  # isort:skip  # noqa: F401

import vcr_langchain as vcr
from langchain.agents import initialize_agent, load_tools
from langchain.llms import OpenAI

import langchain_contrib.tools  # noqa: F401


@vcr.use_cassette()
async def test_use_terminal() -> str:
    """Check that the agent can use the terminal.

    This should expose the terminal's statefulness in a way that the regular
    `BashProcess` does not support.
    """
    llm = OpenAI(temperature=0)
    tools = load_tools(["persistent_terminal"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True
    )
    result = agent.run(
        "List the folders in the current directory. Enter into one of them. List "
        "folders again."
    )
    assert "After entering into the langchain_contrib folder" in result
    return result


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(test_use_terminal)
