"""Module for defining a single iteration of the MRKL agent loop."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseLanguageModel
from langchain.tools.base import BaseTool

from langchain_contrib.chains import ChoiceChain, ToolChain
from langchain_contrib.chains.testing import FakePicker

from .pick_action import MrklPickActionChain


class MrklLoopChain(ChoiceChain):
    """Chain executing one single iteration of the MRKL agent."""

    emit_io_info: bool = True

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        prompt: Optional[BasePromptTemplate] = None,
        embed_scratchpad: bool = True,
        **kwargs: Any,
    ) -> MrklLoopChain:
        """Create a new instance of the chain from tools."""
        picker = MrklPickActionChain.from_llm_and_tools(
            llm, tools, prompt, embed_scratchpad
        )
        choices: Dict[str, Chain] = {tool.name: ToolChain(tool=tool) for tool in tools}
        choices[picker.final_answer_action] = FakePicker(
            input_key="action_input", output_key="answer"
        )
        return cls(
            choice_picker=picker, choices=choices, ignore_keys=["thought"], **kwargs
        )
