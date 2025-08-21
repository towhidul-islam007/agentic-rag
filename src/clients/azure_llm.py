"""Azure OpenAI LLM client implementation"""

import logging

from typing import Any

from clients.base import BaseLLM

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class AzureOpenAILLMWrapper(BaseLLM):
    """Wrapper for Azure OpenAI LLM"""

    def __init__(
        self,
        deployment_name: str = "gpt-4",
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the Azure OpenAI LLM wrapper.

        Args:
            deployment_name: The Azure OpenAI deployment name to use.
            temperature: The temperature setting for the LLM (0.0 to 1.0).
            **kwargs: Additional keyword arguments passed to the LLM constructor.

        Raises:
            ImportError: If langchain-openai is not installed.
        """
        self.deployment_name = deployment_name
        self.temperature = temperature
        self.kwargs = kwargs
        self._is_structured = False  # Track if this is a structured output LLM
        self._llm = self._create_llm()

    def _create_llm(self) -> Any:
        """Create the underlying Azure OpenAI LLM instance.

        Returns:
            AzureChatOpenAI: The configured LLM instance.

        Raises:
            ImportError: If langchain-openai is not installed.
        """
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as e:
            msg = (
                "langchain-openai is required for Azure OpenAI support. "
                "Install it with: pip install langchain-openai"
            )
            raise ImportError(msg) from e

        # Build LLM config
        llm_config = {
            "azure_deployment": self.deployment_name,
            "api_version": settings.azure_openai_api_version,
            "temperature": self.temperature,
            **self.kwargs,
        }

        # Add required Azure OpenAI parameters
        if settings.azure_openai_api_key:
            llm_config["api_key"] = settings.azure_openai_api_key

        if settings.azure_openai_endpoint:
            llm_config["azure_endpoint"] = settings.azure_openai_endpoint

        return AzureChatOpenAI(**llm_config)

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

    def with_structured_output(self, schema: Any) -> "AzureOpenAILLMWrapper":
        """Configure LLM to return structured output.

        Args:
            schema: Output schema definition.

        Returns:
            AzureOpenAILLMWrapper: LLM instance configured for structured output.
        """
        # Create a new instance with structured output
        structured_llm = self._llm.with_structured_output(schema)

        # Create a new wrapper instance
        new_wrapper = AzureOpenAILLMWrapper(
            deployment_name=self.deployment_name,
            temperature=self.temperature,
            **self.kwargs,
        )
        new_wrapper.configure_as_structured(structured_llm)
        return new_wrapper
