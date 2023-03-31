"""Module defining a more flexible BasePromptTemplate."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Union

from langchain.prompts.base import (
    BasePromptTemplate,
    StringPromptTemplate,
    StringPromptValue,
    check_valid_template,
)
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import PromptValue
from pydantic import Field, root_validator


class ZBasePromptTemplate(BasePromptTemplate):
    """A prompt template class that allows for arbitrary partials."""

    base_template: Optional[BasePromptTemplate] = None
    """The actual template that this class wraps around.

    If None, then this class is assumed to be overridden.
    """
    permissive_partial_variables: Mapping[str, Any] = Field(default_factory=dict)
    """Partial variables of any type.

    The BasePromptTemplate.format and format_prompt functions take in any arbitrary
    types, so why shouldn't partials as well?
    """

    @classmethod
    def from_base_template(
        cls, base_template: BasePromptTemplate, **kwargs: Any
    ) -> ZBasePromptTemplate:
        """Wrap around a base template."""
        return cls(
            base_template=base_template,
            input_variables=base_template.input_variables,
            **kwargs,
        )

    def format(self, **kwargs: Any) -> str:
        """Format prompt template as a string."""
        return self.format_prompt(**kwargs).to_string()

    def _format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format the prompt with partials taken care of."""
        raise NotImplementedError(
            "Either override _format_prompt or supply a base template"
        )

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format the prompt from the base prompt."""
        new_kwargs = self._merge_partial_and_user_variables(**kwargs)
        if self.base_template:
            return self.base_template.format_prompt(**new_kwargs)
        else:
            return self._format_prompt(**new_kwargs)

    @property
    def _prompt_type(self) -> str:
        """Return the type of prompt this is."""
        assert (
            self.base_template is not None
        ), "Either override _prompt_type or supply a base template"
        return self.base_template._prompt_type

    def partial(self, **kwargs: Union[str, Callable[[], str]]) -> ZBasePromptTemplate:
        """Return a partial of the prompt template."""
        prompt_dict = self.__dict__.copy()
        prompt_dict["input_variables"] = list(
            set(self.input_variables).difference(kwargs)
        )
        prompt_dict["permissive_partial_variables"] = {
            **self.permissive_partial_variables,
            **kwargs,
        }
        return type(self)(**prompt_dict)

    def permissive_partial(self, **kwargs: Any) -> ZBasePromptTemplate:
        """Return a partial of the prompt template.

        Permissive version that allows for arbitrary input types.
        """
        prompt_dict = self.__dict__.copy()
        prompt_dict["input_variables"] = list(
            set(self.input_variables).difference(kwargs)
        )
        prompt_dict["permissive_partial_variables"] = {
            **self.permissive_partial_variables,
            **kwargs,
        }
        return type(self)(**prompt_dict)

    def _partial_to_str(self, key: str, partial: Any) -> str:
        """Convert a partial value of any type into a str.

        This takes in a key as well to allow for key-based partial conversions.
        """
        if isinstance(partial, str):
            return partial
        elif callable(partial):
            return partial()
        else:
            return str(partial)

    def _combined_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """Combine all kwargs into one dict."""
        return {
            **self.partial_variables,
            **self.permissive_partial_variables,
            **kwargs,
        }

    def _merge_partial_and_user_variables(self, **kwargs: Any) -> Dict[str, Any]:
        """Merge all partials, including permissive ones."""
        combined_kwargs = self._combined_kwargs(**kwargs)
        str_kwargs = {k: self._partial_to_str(k, v) for k, v in combined_kwargs.items()}
        return str_kwargs


class ZStringPromptTemplate(ZBasePromptTemplate, StringPromptTemplate):
    """A version of StringPromptTemplate with extended flexibility."""


class ZPromptTemplate(ZBasePromptTemplate, PromptTemplate):
    """A version of PromptTemplate with extended flexibility."""

    def _format_prompt(self, **kwargs: Any) -> PromptValue:
        return StringPromptValue(text=PromptTemplate.format(self, **kwargs))

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        """Check that template and input variables are consistent."""
        if values["validate_template"]:
            all_inputs = (
                values["input_variables"]
                + list(values["partial_variables"])
                + list(values["permissive_partial_variables"])
            )
            check_valid_template(
                values["template"], values["template_format"], all_inputs
            )
        return values


class ZChatPromptTemplate(ZBasePromptTemplate, ChatPromptTemplate):
    """A version of ChatPromptTemplate with extended flexibility."""

    def _format_prompt(self, **kwargs: Any) -> PromptValue:
        return ChatPromptTemplate.format_prompt(self, **kwargs)

    def partial(self, **kwargs: Union[str, Callable[[], str]]) -> ZBasePromptTemplate:
        """Return a partial of the chat prompt template."""
        return ZBasePromptTemplate.partial(self, **kwargs)
