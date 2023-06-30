"""Module that adds safety to the terminal."""

from typing import Any, Dict, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.tools.base import BaseTool
from pydantic import Field

from langchain_contrib.chains import ToolChain
from langchain_contrib.llms import Human
from langchain_contrib.prompts import ChoicePromptTemplate

from .tool import TerminalTool


class TerminalToolChain(ToolChain):
    """Chain for running the Terminal."""

    tool: BaseTool = Field(default_factory=TerminalTool)
    tool_input_key: str = "command"
    tool_output_key: str = "output"


class SafeTerminalChain(Chain):
    """Chain that lets the human review terminal output before execution."""

    human: BaseLanguageModel = Field(default_factory=Human)
    """The LLM that will review this command for safety."""
    terminal: BaseTool = Field(default_factory=TerminalTool)
    """The terminal that this will use after human review."""
    proceed_choice: str = "Proceed"
    edit_command_choice: str = "Edit command"

    @property
    def input_keys(self) -> List[str]:
        """Input keys this chain expects."""
        return ["command"]

    @property
    def output_keys(self) -> List[str]:
        """Output keys this chain expects."""
        return ["command", "output", "overrides"]

    @property
    def review_prompt(self) -> BasePromptTemplate:
        """Prompt the user for decision making."""
        return ChoicePromptTemplate.from_template(
            """
The LLM would like to run the command `{command}`. You can choose to {choices}.

Your choice: """.lstrip(),
        ).permissive_partial(choices=[self.proceed_choice, self.edit_command_choice])

    @property
    def edit_command_prompt(self) -> BasePromptTemplate:
        """Prompt the user for the new command."""
        return PromptTemplate.from_template("Replace it with: ")

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        pick_action = LLMChain(
            llm=self.human, prompt=self.review_prompt, output_key="choice"
        )
        run_terminal = TerminalToolChain(tool=self.terminal)

        action = pick_action(inputs)["choice"]
        overrides = {}
        if action == self.proceed_choice:
            command = inputs["command"]
        else:
            edit_command = LLMChain(
                llm=self.human, prompt=self.edit_command_prompt, output_key="command"
            )
            overrides["command"] = inputs["command"]
            command = edit_command(inputs)["command"]

        results = run_terminal({"command": command})
        return {
            **results,
            "overrides": overrides,
        }
