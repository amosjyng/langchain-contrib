"""Module that defines the choice prompt."""
from __future__ import annotations

from typing import Any, Callable, List, Sequence, Union

from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import BaseMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseMessage
from pydantic import Field

from langchain_contrib.utils import f_join

from .prompt_value import BaseChoicePrompt

ChoicesFormatter = Callable[[List[str]], str]


def get_simple_joiner(joiner: str = ", ") -> ChoicesFormatter:
    """Get a choice formatter that's just a simple joining of strings."""

    def simple_join_choice(choices: List[str]) -> str:
        """Do a simple join on the choice strings."""
        return f_join(joiner, choices)

    return simple_join_choice


def get_oxford_comma_formatter(conjunction: str = "or") -> ChoicesFormatter:
    """Get a choice formatter that respects the Oxford comma."""

    def oxford_comma_list(choices: List[str]) -> str:
        """Phrase the list using the Oxford comma."""
        if len(choices) == 0:
            return ""
        elif len(choices) == 1:
            return choices[0]
        elif len(choices) == 2:
            return f_join(f" {conjunction} ", choices)
        else:
            head = f_join(", ", choices[:-1])
            return f_join(f", {conjunction} ", [head, choices[-1]])

    return oxford_comma_list


def list_of_choices(choices: List[str]) -> str:
    """Return a numerical list of choices."""
    return f_join("\n", [f"{i+1}. {choice}" for i, choice in enumerate(choices)])


class ChoicePromptTemplate(BasePromptTemplate):
    """A wrapper prompt template for picking from a number of choices.

    This template preserves choice information in prompts.
    """

    base_template: BasePromptTemplate
    """The base template that this class wraps around."""
    choices: List[str]
    """The list of choices to pick from."""
    choices_formatter: ChoicesFormatter = Field(
        default_factory=get_oxford_comma_formatter
    )
    """How to convert from the list of choices to a single string.

    Utility functions to help with this include:

    - get_simple_joiner
    - get_oxford_comma_formatter
    - list_of_choices
    """
    choice_format_key: str = "choices"
    """Which string is used for formatting choices in the template."""

    @classmethod
    def from_base_template(
        cls, base_template: BasePromptTemplate, choices: List[str], **kwargs: Any
    ) -> ChoicePromptTemplate:
        """Load a ChoicePromptTemplate from base templates."""
        result = cls(
            base_template=base_template,
            input_variables=base_template.input_variables,
            choices=choices,
            **kwargs,
        )

        result.partial_variables = {
            **result.partial_variables,
            result.choice_format_key: result._no_args_formatter,
        }
        result.input_variables.remove(result.choice_format_key)
        return result

    @classmethod
    def from_template(
        cls, template: str, choices: List[str], **kwargs: Any
    ) -> ChoicePromptTemplate:
        """Load a ChoicePromptTemplate from a text template."""
        base_template = PromptTemplate.from_template(template)
        return cls.from_base_template(
            base_template=base_template, choices=choices, **kwargs
        )

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[Union[BaseMessagePromptTemplate, BaseMessage]],
        choices: List[str],
        **kwargs: Any,
    ) -> ChoicePromptTemplate:
        """Load a ChoicePromptTemplate from message templates."""
        base_template = ChatPromptTemplate.from_messages(messages)
        return cls.from_base_template(
            base_template=base_template, choices=choices, **kwargs
        )

    @property
    def _prompt_type(self) -> str:
        return "choice"

    def _no_args_formatter(self) -> str:
        """No-argument choice formatter for partials."""
        return self.choices_formatter(self.choices)

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs."""
        return self.format_prompt(**kwargs).to_string()

    def format_prompt(self, **kwargs: Any) -> BaseChoicePrompt:
        """Format the prompt while preserving the choices."""
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        prompt = self.base_template.format_prompt(**kwargs)
        return BaseChoicePrompt.from_prompt(prompt, choices=self.choices)
