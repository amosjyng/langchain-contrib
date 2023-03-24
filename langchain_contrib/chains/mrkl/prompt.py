"""Prompting configuration for MRKL agents."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from fvalues import F
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate
from langchain.tools.base import BaseTool
from pydantic import BaseModel, Extra

from langchain_contrib.prompts import ChoicePromptTemplate
from langchain_contrib.prompts.choice import get_simple_joiner
from langchain_contrib.utils import f_join


class BaseMrklPrompt(BaseModel, ABC):
    """A base class for MRKL prompting."""

    tools: List[BaseTool]
    """Tools that the MRKL agent has access to."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    @abstractmethod
    def base_prompt(self) -> BasePromptTemplate:
        """The basis for the MRKL prompt."""

    @property
    def tool_names(self) -> List[str]:
        """Just the names of the tools, without their descriptions."""
        return [tool.name for tool in self.tools]

    @property
    def tool_descriptions(self) -> F:
        """Descriptions of the tools and their usage."""
        return f_join(
            "\n", [F(f"{tool.name}: {tool.description}") for tool in self.tools]
        )

    def template(self) -> ChoicePromptTemplate:
        """Return a ChoicePromptTemplate for these tools."""
        partial_choice = ChoicePromptTemplate.from_base_template(
            base_template=self.base_prompt,
            choices=self.tool_names,
            choice_format_key="tool_names",
            choices_formatter=get_simple_joiner(),
        ).partial(
            tool_descriptions=self.tool_descriptions,
        )
        assert isinstance(partial_choice, ChoicePromptTemplate)
        return partial_choice


class StringMrklPrompt(BaseMrklPrompt):
    """The string version of the prompt to the MRKL agent."""

    template_text: str = """
Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""".lstrip()  # noqa

    @property
    def base_prompt(self) -> BasePromptTemplate:
        """The basis for the string MRKL prompt."""
        return PromptTemplate.from_template(self.template_text)


class ChatMrklPrompt(BaseMrklPrompt):
    """The chat version of the prompt to the MRKL agent."""

    system_template: str = """
Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are: {tool_names}

The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

Here is an example of an invalid $JSON_BLOB:

```
{{{{
  "action": $FIRST_TOOL_NAME,
  "action_input": $FIRST_INPUT
}}}}

{{{{
  "action": $SECOND_TOOL_NAME,
  "action_input": $SECOND_INPUT
}}}}
```

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Reminder to always use the exact characters `Final Answer` when responding.
""".strip()  # noqa

    human_template: str = """{input}\n\n{agent_scratchpad}"""

    @property
    def base_prompt(self) -> BasePromptTemplate:
        """The basis for the chat MRKL prompt."""
        return ChatPromptTemplate.from_messages(
            messages=[
                SystemMessagePromptTemplate.from_template(self.system_template),
                HumanMessagePromptTemplate.from_template(self.human_template),
            ]
        )


class MrklPromptSelector(ConditionalPromptSelector):
    """Prompt definitions for the MRKL agent."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def from_tools(cls, tools: List[BaseTool]) -> MrklPromptSelector:
        """Construct MRKL prompt selector from a list of tools."""
        string_prompt_template = StringMrklPrompt(tools=tools).template()
        chat_prompt_template = ChatMrklPrompt(tools=tools).template()
        return cls(
            default_prompt=string_prompt_template,
            conditionals=[(is_chat_model, chat_prompt_template)],
        )
