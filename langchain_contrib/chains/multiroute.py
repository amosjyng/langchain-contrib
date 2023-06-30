"""Module for a custom MultiRouteChain."""
from __future__ import annotations

from typing import Any, List, Optional

from langchain.chains.base import Chain
from langchain.chains.router import MultiRouteChain, RouterChain
from langchain.input import get_color_mapping
from langchain.tools.base import BaseTool

from langchain_contrib.chains import DummyLLMChain

from .tool import ToolChain


class ZMultiRouteChain(MultiRouteChain):
    """A multi-route chain that can be used with tools."""

    @classmethod
    def from_tools(
        cls,
        router_chain: RouterChain,
        tools: List[BaseTool],
        default_chain: Optional[Chain] = None,
        excluded_colors: List[str] = ["green"],
        verbose: bool = False,
        **kwargs: Any,
    ) -> ZMultiRouteChain:
        """Construct a ZMultiRouteChain from tools.

        This also assigns colors to each tool.
        """
        from langchain_contrib.tools import ZBaseTool

        if default_chain is None:
            default_chain = DummyLLMChain()

        color_mapping = get_color_mapping(
            items=[str(i) for i in range(len(tools))],
            excluded_colors=excluded_colors,
        )
        wrapped_tools = [
            ZBaseTool.from_tool(
                base_tool=tool, color=color_mapping[str(i)], verbose=verbose
            )
            for i, tool in enumerate(tools)
        ]
        choices = {
            tool.name: ToolChain(tool=tool, verbose=False) for tool in wrapped_tools
        }
        return cls(
            router_chain=router_chain,
            default_chain=default_chain,
            destination_chains=choices,
            verbose=verbose,
            **kwargs,
        )
