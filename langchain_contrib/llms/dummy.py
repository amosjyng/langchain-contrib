"""A module for defining a dummy LLM."""

from typing import List, Optional

from langchain.schema import BaseLanguageModel, LLMResult, PromptValue


class DummyLanguageModel(BaseLanguageModel):
    """A dummy LLM for when you need an LLM but don't care for a real one.

    You can use this instead of FakeLLM when you want to be sure the LLM is not
    actually getting called.
    """

    def generate_prompt(
        self, prompts: List[PromptValue], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Error out because this is a dummy LLM."""
        raise NotImplementedError("You're using the dummy LLM")

    async def agenerate_prompt(
        self, prompts: List[PromptValue], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Error out asynchronously because this is a dummy LLM."""
        raise NotImplementedError("You're using the dummy LLM")
