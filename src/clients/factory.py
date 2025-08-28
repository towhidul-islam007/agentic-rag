"""Client factory for managing vector stores and embedders"""

import logging

from enum import Enum
from typing import Any, Optional

from clients.base import BaseDocumentStore, BaseEmbedder, BaseLLM, BaseRetriever
from clients.chroma_client import ChromaDocumentStoreWrapper, ChromaRetriever
from clients.google_embedder import GoogleEmbedder
from clients.google_llm import GoogleLLMWrapper

from config import get_settings

logger = logging.getLogger(__name__)


class VectorStoreType(Enum):
    """Supported vector store types"""

    CHROMA = "chroma"
    ELASTICSEARCH = "elasticsearch"


class EmbedderType(Enum):
    """Supported embedder types"""

    GOOGLE = "google"


class LLMProvider(Enum):
    """Supported LLM providers"""

    GOOGLE = "google"
    AZURE = "azure"


class ClientFactory:
    """Factory for creating vector store, embedder, and LLM clients."""

    def __init__(self) -> None:
        """Initialize the client factory."""
        self.settings = get_settings()
        self._document_stores = {}
        self._retrievers = {}
        self._embedders = {}
        self._llms = {}
        self._embedders = {}

    def get_embedder(
        self,
        embedder_type: EmbedderType = EmbedderType.GOOGLE,
        model: Optional[str] = None,
    ) -> BaseEmbedder:
        """Get or create an embedder instance.

        Args:
            embedder_type: Type of embedder to create. Defaults to Google.
            model: Model name to use. If None, uses default.

        Returns:
            BaseEmbedder: The result.
        """
        cache_key = f"{embedder_type.value}_{model or 'default'}"

        if cache_key not in self._embedders:
            if embedder_type == EmbedderType.GOOGLE:
                self._embedders[cache_key] = GoogleEmbedder(
                    model=model or "models/embedding-001"
                )
            else:
                msg = f"Unsupported embedder type: {embedder_type}"
                raise ValueError(msg)

        return self._embedders[cache_key]

    def get_document_store(
        self, store_type: VectorStoreType = VectorStoreType.CHROMA, **kwargs: Any
    ) -> BaseDocumentStore:
        """Get or create a document store instance.

        Args:
            store_type: The store_type parameter. Defaults to VectorStoreType.CHROMA.
            **kwargs: Additional keyword arguments passed to the document store
                constructor.

        Returns:
            BaseDocumentStore: The result.
        """
        cache_key = f"{store_type.value}_{hash(frozenset(kwargs.items()))}"

        if cache_key not in self._document_stores:
            if store_type == VectorStoreType.CHROMA:
                self._document_stores[cache_key] = ChromaDocumentStoreWrapper(
                    collection_name=kwargs.get("collection_name", "documents")
                )
            elif store_type == VectorStoreType.ELASTICSEARCH:
                # Import here to avoid dependency issues if ES is not configured
                from clients.elasticsearch_client import (
                    ElasticsearchDocumentStoreWrapper,
                )

                self._document_stores[cache_key] = ElasticsearchDocumentStoreWrapper(
                    index=kwargs.get("index")
                )
            else:
                msg = f"Unsupported document store type: {store_type}"
                raise ValueError(msg)

        return self._document_stores[cache_key]

    def get_retriever(
        self,
        store_type: VectorStoreType = VectorStoreType.CHROMA,
        document_store: Optional[BaseDocumentStore] = None,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> BaseRetriever:
        """Get or create a retriever instance.

        Args:
            store_type: The store_type parameter. Defaults to VectorStoreType.CHROMA.
            document_store: Document to process. If None, uses default.
            top_k: Maximum number of items to return. If None, uses default.
            **kwargs: Additional keyword arguments passed to the document store
                constructor.

        Returns:
            BaseRetriever: The result.
        """
        if document_store is None:
            document_store = self.get_document_store(store_type, **kwargs)

        cache_key = f"{store_type.value}_retriever_{top_k or 'default'}"

        if cache_key not in self._retrievers:
            if store_type == VectorStoreType.CHROMA:
                self._retrievers[cache_key] = ChromaRetriever(
                    document_store=document_store, top_k=top_k
                )
            elif store_type == VectorStoreType.ELASTICSEARCH:
                # Import here to avoid dependency issues if ES is not configured
                from clients.elasticsearch_client import ElasticsearchRetriever

                self._retrievers[cache_key] = ElasticsearchRetriever(
                    document_store=document_store, top_k=top_k
                )
            else:
                msg = f"Unsupported retriever type: {store_type}"
                raise ValueError(msg)

        return self._retrievers[cache_key]

    def get_preferred_store_type(self) -> VectorStoreType:
        """Get the preferred vector store type based on configuration.

        Returns:
            VectorStoreType: The result.
        """
        # Prefer Elasticsearch if configured, otherwise use ChromaDB
        if self.settings.elasticsearch_url and self.settings.elasticsearch_api_key:
            return VectorStoreType.ELASTICSEARCH
        return VectorStoreType.CHROMA

    def get_llm(
        self,
        provider: LLMProvider = LLMProvider.GOOGLE,
        model: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> BaseLLM:
        """Get or create an LLM instance.

        Args:
            provider: The LLM provider to use. Defaults to GOOGLE.
            model: The model name to use. If None, uses provider default.
            temperature: The temperature setting for the LLM (0.0 to 1.0).
            **kwargs: Additional keyword arguments passed to the LLM constructor.

        Returns:
            BaseLLM: The configured LLM instance.

        Raises:
            ValueError: If provider is not supported.
        """
        cache_key = f"{provider.value}_{model or 'default'}_{temperature}"

        if cache_key not in self._llms:
            if provider == LLMProvider.GOOGLE:
                if model is None:
                    model = self.settings.default_llm_model
                self._llms[cache_key] = GoogleLLMWrapper(
                    model=model, temperature=temperature, **kwargs
                )
            elif provider == LLMProvider.AZURE:
                if model is None:
                    model = self.settings.azure_openai_deployment_name
                from clients.azure_llm import AzureOpenAILLMWrapper

                self._llms[cache_key] = AzureOpenAILLMWrapper(
                    deployment_name=model, temperature=temperature, **kwargs
                )
            else:
                msg = f"Unsupported LLM provider: {provider}"
                raise ValueError(msg)

        return self._llms[cache_key]

    def get_preferred_llm_provider(self) -> LLMProvider:
        """Get the preferred LLM provider based on configuration.

        Returns:
            LLMProvider: The preferred LLM provider (always Google).
        """
        return LLMProvider.GOOGLE

    def clear_cache(self) -> None:
        """Clear all cached instances."""
        self._document_stores.clear()
        self._retrievers.clear()
        self._embedders.clear()
        self._llms.clear()
        logger.info("Client factory cache cleared")


# Global factory instance
_factory = None


def get_factory() -> ClientFactory:
    """Get the global client factory instance.

    Returns:
        ClientFactory: The result.
    """
    global _factory
    if _factory is None:
        _factory = ClientFactory()
    return _factory


# Convenience functions
def get_embedder(
    embedder_type: EmbedderType = EmbedderType.GOOGLE, model: Optional[str] = None
) -> BaseEmbedder:
    """Get an embedder instance.

    Args:
        embedder_type: The embedder_type parameter. Defaults to EmbedderType.GOOGLE.
        model: Model name or configuration. If None, uses default.

    Returns:
        BaseEmbedder: The result.
    """
    return get_factory().get_embedder(embedder_type, model)


def get_document_store(
    store_type: Optional[VectorStoreType] = None, **kwargs: Any
) -> BaseDocumentStore:
    """Get a document store instance.

    Args:
        store_type: The store_type parameter. If None, uses default.
        **kwargs: Additional keyword arguments passed to the document store
            constructor.

    Returns:
        BaseDocumentStore: The result.
    """
    factory = get_factory()
    if store_type is None:
        store_type = factory.get_preferred_store_type()
    return factory.get_document_store(store_type, **kwargs)


def get_retriever(
    store_type: Optional[VectorStoreType] = None,
    document_store: Optional[BaseDocumentStore] = None,
    top_k: Optional[int] = None,
    **kwargs: Any,
) -> BaseRetriever:
    """Get a retriever instance.

    Args:
        store_type: The store_type parameter. If None, uses default.
        document_store: Document to process. If None, uses default.
        top_k: Maximum number of items to return. If None, uses default.
        **kwargs: Additional keyword arguments passed to the document store
            constructor.

    Returns:
        BaseRetriever: The result.
    """
    factory = get_factory()
    if store_type is None:
        store_type = factory.get_preferred_store_type()
    return factory.get_retriever(store_type, document_store, top_k, **kwargs)


def get_llm(
    provider: Optional[LLMProvider] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> BaseLLM:
    """Get an LLM instance.

    Args:
        provider: The LLM provider to use. If None, uses default from settings.
        model: The model name to use. If None, uses provider default.
        temperature: The temperature setting for the LLM (0.0 to 1.0).
        **kwargs: Additional keyword arguments passed to the LLM constructor.

    Returns:
        BaseLLM: The configured LLM instance.
    """
    factory = get_factory()
    if provider is None:
        provider = factory.get_preferred_llm_provider()
    return factory.get_llm(provider, model, temperature, **kwargs)
