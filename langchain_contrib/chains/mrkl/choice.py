"""Module for defining a single iteration of the MRKL agent loop."""
from __future__ import annotations

from typing import Any, List, Type

from langchain.schema import BaseLanguageModel
from langchain.tools.base import BaseTool

from langchain_contrib.chains import ChoiceChain, ToolChain

from .pick_action import MrklPickActionChain
from .prompt import MrklPromptSelector


class MrklLoopChain(ChoiceChain):
    """Chain executing one single iteration of the MRKL agent."""

    @classmethod
    def from_tools(
        cls,
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        prompt_selector: Type[MrklPromptSelector] = MrklPromptSelector,
        **kwargs: Any,
    ) -> MrklLoopChain:
        """Create a new instance of the chain from tools."""
        picker = MrklPickActionChain.from_tools(llm, tools, prompt_selector)
        choices = {tool.name: ToolChain(tool=tool) for tool in tools}
        return cls(
            choice_picker=picker, choices=choices, ignore_keys=["observation"], **kwargs
        )
