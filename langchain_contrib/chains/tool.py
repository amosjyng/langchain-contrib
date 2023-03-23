"""Module that defines a Chain wrapper for Tools."""

from typing import Dict, List

from langchain.chains.base import Chain
from langchain.tools.base import BaseTool


class ToolChain(Chain):
    """Wraps a Tool in a Chain.

    Allows for tool and chain interop outside of the Agent class.
    """

    tool: BaseTool
    """The tool this chain will call when invoked."""
    tool_input_key: str = "action_input"
    """The key for this class to feed into the tool."""
    tool_output_key: str = "action_result"
    """The key produced by this class as a result of using the tool."""

    @property
    def input_keys(self) -> List[str]:
        """Input keys this chain expects."""
        return [self.tool_input_key]

    @property
    def output_keys(self) -> List[str]:
        """Output keys this chain expects."""
        return [self.tool_output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Run the tool for this chain."""
        tool_input = inputs[self.tool_input_key]
        tool_output = self.tool.run(tool_input)
        return {self.tool_output_key: tool_output}
