"""LLM factory for creating different types of language models"""

from enum import Enum
from typing import Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from config import get_settings

settings = get_settings()


class LLMProvider(Enum):
    """Supported LLM providers"""

    GOOGLE = "google"
    AZURE = "azure"


class LLMFactory:
    """Factory for creating LLM instances based on provider and configuration"""

    def __init__(self) -> None:
        """Initialize the LLM factory.

        Args:
            None
        """
        self.settings = settings

    def create_llm(
        self,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        structured_output: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Create an LLM instance based on the provider.

        Args:
            provider: The LLM provider to use. If None, uses settings default.
            model: The model name to use. If None, uses provider default.
            temperature: The temperature setting for the LLM (0.0 to 1.0).
            structured_output: Optional structured output schema for the LLM.
            **kwargs: Additional keyword arguments passed to the LLM constructor.

        Returns:
            Any: Configured LLM instance.

        Raises:
            ValueError: If provider is not supported or configuration is invalid.
        """
        if provider is None:
            provider = LLMProvider(self.settings.llm_provider)

        if provider == LLMProvider.GOOGLE:
            return self._create_google_llm(
                model, temperature, structured_output, **kwargs
            )
        if provider == LLMProvider.AZURE:
            return self._create_azure_llm(
                model, temperature, structured_output, **kwargs
            )

        msg = f"Unsupported LLM provider: {provider}"
        raise ValueError(msg)

    def _create_google_llm(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        structured_output: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a Google Generative AI LLM instance.

        Args:
            model: The model name to use. If None, uses default from settings.
            temperature: The temperature setting for the LLM (0.0 to 1.0).
            structured_output: Optional structured output schema for the LLM.
            **kwargs: Additional keyword arguments passed to the LLM constructor.

        Returns:
            Any: Configured Google LLM instance.
        """
        if model is None:
            model = self.settings.default_llm_model

        # Build LLM config
        llm_config = {"model": model, "temperature": temperature, **kwargs}

        # Add API key if available
        if self.settings.google_api_key:
            llm_config["google_api_key"] = self.settings.google_api_key

        llm = ChatGoogleGenerativeAI(**llm_config)

        if structured_output:
            return llm.with_structured_output(structured_output)
        return llm

    def _create_azure_llm(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        structured_output: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Create an Azure OpenAI LLM instance.

        Args:
            model: The model/deployment name to use. If None, uses default
                from settings.
            temperature: The temperature setting for the LLM (0.0 to 1.0).
            structured_output: Optional structured output schema for the LLM.
            **kwargs: Additional keyword arguments passed to the LLM constructor.

        Returns:
            Any: Configured Azure OpenAI LLM instance.

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

        if model is None:
            model = self.settings.azure_openai_deployment_name

        # Build LLM config
        llm_config = {
            "azure_deployment": model,
            "api_version": self.settings.azure_openai_api_version,
            "temperature": temperature,
            **kwargs,
        }

        # Add required Azure OpenAI parameters
        if self.settings.azure_openai_api_key:
            llm_config["api_key"] = self.settings.azure_openai_api_key

        if self.settings.azure_openai_endpoint:
            llm_config["azure_endpoint"] = self.settings.azure_openai_endpoint

        llm = AzureChatOpenAI(**llm_config)

        if structured_output:
            return llm.with_structured_output(structured_output)
        return llm

    def get_supported_providers(self) -> list[str]:
        """Get list of supported LLM providers.

        Returns:
            list[str]: List of supported provider names.
        """
        return [provider.value for provider in LLMProvider]


# Global factory instance
_llm_factory = LLMFactory()


def create_llm(
    provider: Optional[LLMProvider] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    structured_output: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Create an LLM instance using the global factory.

    Args:
        provider: The LLM provider to use. If None, uses settings default.
        model: The model name to use. If None, uses provider default.
        temperature: The temperature setting for the LLM (0.0 to 1.0).
        structured_output: Optional structured output schema for the LLM.
        **kwargs: Additional keyword arguments passed to the LLM constructor.

    Returns:
        Any: Configured LLM instance.
    """
    return _llm_factory.create_llm(
        provider=provider,
        model=model,
        temperature=temperature,
        structured_output=structured_output,
        **kwargs,
    )


def get_llm_factory() -> LLMFactory:
    """Get the global LLM factory instance.

    Returns:
        LLMFactory: The global LLM factory instance.
    """
    return _llm_factory
