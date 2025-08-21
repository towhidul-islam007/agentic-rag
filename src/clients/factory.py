"""Client factory for managing vector stores and embedders"""

import logging

from enum import Enum
from typing import Any, Optional

from config import get_settings
from clients.base import BaseDocumentStore, BaseEmbedder, BaseRetriever
from clients.chroma_client import ChromaDocumentStoreWrapper, ChromaRetriever
from clients.google_embedder import GoogleEmbedder

logger = logging.getLogger(__name__)


class VectorStoreType(Enum):
    """Supported vector store types"""

    CHROMA = "chroma"
    ELASTICSEARCH = "elasticsearch"


class EmbedderType(Enum):
    """Supported embedder types"""

    GOOGLE = "google"


class ClientFactory:
    """Factory for creating vector store and embedder clients"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._document_stores = {}
        self._retrievers = {}
        self._embedders = {}

    def get_embedder(
        self,
        embedder_type: EmbedderType = EmbedderType.GOOGLE,
        model: Optional[str] = None,
    ) -> BaseEmbedder:
        """Get or create an embedder instance"""
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
        """Get or create a document store instance"""
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
        """Get or create a retriever instance"""
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
        """Get the preferred vector store type based on configuration"""
        # Prefer Elasticsearch if configured, otherwise use ChromaDB
        if self.settings.elasticsearch_url and self.settings.elasticsearch_api_key:
            return VectorStoreType.ELASTICSEARCH
        return VectorStoreType.CHROMA

    def clear_cache(self) -> None:
        """Clear all cached instances"""
        self._document_stores.clear()
        self._retrievers.clear()
        self._embedders.clear()
        logger.info("Client factory cache cleared")


# Global factory instance
_factory = None


def get_factory() -> ClientFactory:
    """Get the global client factory instance"""
    global _factory
    if _factory is None:
        _factory = ClientFactory()
    return _factory


# Convenience functions
def get_embedder(
    embedder_type: EmbedderType = EmbedderType.GOOGLE, model: Optional[str] = None
) -> BaseEmbedder:
    """Get an embedder instance"""
    return get_factory().get_embedder(embedder_type, model)


def get_document_store(
    store_type: Optional[VectorStoreType] = None, **kwargs: Any
) -> BaseDocumentStore:
    """Get a document store instance"""
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
    """Get a retriever instance"""
    factory = get_factory()
    if store_type is None:
        store_type = factory.get_preferred_store_type()
    return factory.get_retriever(store_type, document_store, top_k, **kwargs)
