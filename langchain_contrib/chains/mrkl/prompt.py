"""Prompting configuration for MRKL agents."""

from typing import List

from fvalues import F
from langchain.chains.prompt_selector import (
    BasePromptSelector,
    ConditionalPromptSelector,
    is_chat_model,
)
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseLanguageModel
from langchain.tools.base import BaseTool
from pydantic import BaseModel, Extra

from langchain_contrib.utils import f_join


class MrklPromptSelector(BaseModel):
    """Prompt definitions for the MRKL agent."""

    tools: List[BaseTool]
    """Tools that the MRKL agent has access to."""

    string_template_text: str = """
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

    chat_system_template_text: str = """
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

    chat_human_template: str = """{input}\n\n{agent_scratchpad}"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def tool_names(self) -> str:
        """Just the names of the tools, without their descriptions."""
        return ", ".join([tool.name for tool in self.tools])

    @property
    def tool_descriptions(self) -> F:
        """Descriptions of the tools and their usage."""
        return f_join(
            "\n", [F(f"{tool.name}: {tool.description}") for tool in self.tools]
        )

    @property
    def string_prompt_template(self) -> PromptTemplate:
        """String prompt template for the MRKL agent."""
        prompt = PromptTemplate.from_template(self.string_template_text).partial(
            tool_names=self.tool_names,
            tool_descriptions=self.tool_descriptions,
        )
        assert isinstance(prompt, PromptTemplate)
        return prompt

    @property
    def chat_prompt_template(self) -> ChatPromptTemplate:
        """Chat prompt template for the MRKL agent."""
        system_string_prompt = PromptTemplate.from_template(
            self.chat_system_template_text
        ).partial(
            tool_names=self.tool_names,
            tool_descriptions=self.tool_descriptions,
        )
        assert isinstance(system_string_prompt, PromptTemplate)
        return ChatPromptTemplate.from_messages(
            messages=[
                SystemMessagePromptTemplate(prompt=system_string_prompt),
                HumanMessagePromptTemplate.from_template(self.chat_human_template),
            ]
        )

    @property
    def prompt_selector(self) -> BasePromptSelector:
        """Prompt selector for the MRKL agent."""
        return ConditionalPromptSelector(
            default_prompt=self.string_prompt_template,
            conditionals=[(is_chat_model, self.chat_prompt_template)],
        )

    def get_prompt(self, llm: BaseLanguageModel) -> BasePromptTemplate:
        """Get the prompt customized for the given LLM."""
        return self.prompt_selector.get_prompt(llm)
