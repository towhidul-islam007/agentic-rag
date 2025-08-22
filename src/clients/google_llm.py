"""Google Generative AI LLM client implementation"""

from __future__ import annotations

import logging

from typing import Any

from clients.base import BaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class GoogleLLMWrapper(BaseLLM):
    """Wrapper for Google Generative AI LLM"""

    def __init__(
        self,
        model: str = "gemini-2.0-flash-exp",
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the Google LLM wrapper.

        Args:
            model: The Google model name to use.
            temperature: The temperature setting for the LLM (0.0 to 1.0).
            **kwargs: Additional keyword arguments passed to the LLM constructor.
        """
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs
        self._is_structured = False  # Track if this is a structured output LLM
        self._llm = self._create_llm()

    def _create_llm(self) -> ChatGoogleGenerativeAI:
        """Create the underlying Google LLM instance.

        Returns:
            ChatGoogleGenerativeAI: The configured LLM instance.
        """
        # Build LLM config
        llm_config = {
            "model": self.model,
            "temperature": self.temperature,
            **self.kwargs,
        }

        # Add API key if available
        if settings.google_api_key:
            llm_config["google_api_key"] = settings.google_api_key

        return ChatGoogleGenerativeAI(**llm_config)

    def invoke(self, prompt: str) -> Any:
        """Synchronously invoke the LLM with a prompt.

        Args:
            prompt: Input prompt string.

        Returns:
            Any: Generated response (str for regular LLM, structured object for
                structured output).
        """
        response = self._llm.invoke(prompt)

        # If this is a structured output LLM, return the response directly
        if self._is_structured:
            return response

        # Otherwise, extract content from the response
        return response.content if hasattr(response, "content") else str(response)

    async def ainvoke(self, prompt: str) -> Any:
        """Asynchronously invoke the LLM with a prompt.

        Args:
            prompt: Input prompt string.

        Returns:
            Any: Generated response (str for regular LLM, structured object for
                structured output).
        """
        response = await self._llm.ainvoke(prompt)

        # If this is a structured output LLM, return the response directly
        if self._is_structured:
            return response

        # Otherwise, extract content from the response
        return response.content if hasattr(response, "content") else str(response)

    def with_structured_output(self, schema: Any) -> GoogleLLMWrapper:
        """Configure LLM to return structured output.

        Args:
            schema: Output schema definition.

        Returns:
            GoogleLLMWrapper: LLM instance configured for structured output.
        """
        # Create a new instance with structured output
        structured_llm = self._llm.with_structured_output(schema)

        # Create a new wrapper instance
        new_wrapper = GoogleLLMWrapper(
            model=self.model, temperature=self.temperature, **self.kwargs
        )
        new_wrapper.configure_as_structured(structured_llm)
        return new_wrapper
