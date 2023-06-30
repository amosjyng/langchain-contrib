"""Module for defining a single iteration of the MRKL agent loop."""
from __future__ import annotations

from typing import Any, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.prompts.base import BasePromptTemplate
from langchain.tools.base import BaseTool

from langchain_contrib.chains import ZMultiRouteChain

from .pick_action import MrklPickActionChain


class MrklLoopChain(ZMultiRouteChain):
    """Chain executing one single iteration of the MRKL agent."""

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
        picker = MrklPickActionChain.from_tools(llm, tools, prompt, embed_scratchpad)
        loop_chain = cls.from_tools(
            router_chain=picker,
            tools=tools,
            **kwargs,
        )
        assert isinstance(loop_chain, MrklLoopChain)
        return loop_chain
