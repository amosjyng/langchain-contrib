"""Types useful for prompting."""

from typing import Union

from langchain.prompts import PromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import BaseMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import BaseMessage

Templatable = Union[str, BaseMessagePromptTemplate, BaseMessage, BasePromptTemplate]
"""Anything that can be converted directly into a BasePromptTemplate."""


def into_template(templatable: Templatable) -> BasePromptTemplate:
    """Convert a Templatable into a proper BasePromptTemplate."""
    if isinstance(templatable, str):
        return PromptTemplate.from_template(templatable)
    elif isinstance(templatable, BaseMessagePromptTemplate) or isinstance(
        templatable, BaseMessage
    ):
        return ChatPromptTemplate.from_messages([templatable])
    elif isinstance(templatable, BasePromptTemplate):
        return templatable
    else:
        raise ValueError(f"Don't know how to convert {templatable} into template")
