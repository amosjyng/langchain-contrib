"""LLM-related utility functions."""

from typing import List, Optional

from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLLM
from langchain.schema import BaseLanguageModel, PromptValue


def call_llm(
    llm: BaseLanguageModel, prompt: PromptValue, stop: Optional[List[str]] = None
) -> str:
    """Invoke the LLM's __call__ function.

    The BaseLanguageModel class does not have that function because the function
    signatures are slightly different for each child class, but you can readily get a
    str back all the same.
    """
    if isinstance(llm, BaseLLM):
        return llm(prompt=prompt.to_string(), stop=stop)
    if isinstance(llm, BaseChatModel):
        return llm(messages=prompt.to_messages(), stop=stop).content
    raise ValueError(f"Unknown type of LLM: {llm}")
