"""Defines the Chained prompt template type."""

from typing import Any, List, Union

from langchain.prompts import PromptTemplate
from langchain.prompts.base import (
    BasePromptTemplate,
    StringPromptTemplate,
    StringPromptValue,
)
from langchain.prompts.chat import (
    BaseMessagePromptTemplate,
    ChatPromptTemplate,
    ChatPromptValue,
)
from langchain.schema import BaseMessage, PromptValue

from langchain_contrib.utils import f_join, safe_inputs

Templatable = Union[str, BaseMessagePromptTemplate, BaseMessage, BasePromptTemplate]
"""Anything that can be converted directly into a BasePromptTemplate."""


class ChainedPromptTemplate(StringPromptTemplate):
    """A prompt template composed of multiple other prompt templates chained together.

    This is a StringPromptTemplate rather than a BasePromptTemplate to enable use in
    BaseStringMessagePromptTemplate.
    """

    joiner: str = ""
    """How to join each template output together.

    Only meaningful for StringPromptTemplate's.
    """
    subprompts: List[BasePromptTemplate]

    def __init__(self, subprompts: List[Templatable], joiner: str = "", **kwargs: Any):
        """Initialize a ChainedPromptTemplate.

        subprompts can be passed in as just plain strings for convenience.
        """
        prompts: List[BasePromptTemplate] = []
        for subprompt in subprompts:
            if isinstance(subprompt, str):
                if subprompt != "":  # ignore empty strings
                    prompts.append(PromptTemplate.from_template(subprompt))
            elif isinstance(subprompt, BaseMessagePromptTemplate) or isinstance(
                subprompt, BaseMessage
            ):
                prompts.append(ChatPromptTemplate.from_messages([subprompt]))
            elif isinstance(subprompt, BasePromptTemplate):
                prompts.append(subprompt)
            else:
                raise ValueError(f"Subprompt {subprompt} has unknown type")
        input_variables = list(
            set([var for subprompt in prompts for var in subprompt.input_variables])
        )
        super().__init__(
            input_variables=input_variables,
            joiner=joiner,  # type: ignore
            subprompts=prompts,  # type: ignore
            **kwargs,
        )

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs."""
        return self.format_prompt(**kwargs).to_string()

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format each series of prompts with the given inputs."""
        unused_args = set(kwargs.keys())
        values: List[PromptValue] = []
        for subprompt in self.subprompts:
            prompt_args = safe_inputs(subprompt, kwargs)
            unused_args -= prompt_args.keys()
            values.append(subprompt.format_prompt(**prompt_args))
        if unused_args:
            raise KeyError(unused_args)

        all_strings = all([isinstance(x, StringPromptValue) for x in values])
        if all_strings:
            text = f_join(self.joiner, [x.to_string() for x in values])
            return StringPromptValue(text=text)
        else:
            messages = [
                message for subvalue in values for message in subvalue.to_messages()
            ]
            return ChatPromptValue(messages=messages)

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        return "chained"
