"""Utility functions for the Agentic RAG system"""

import asyncio
import logging

from typing import List, Optional

from clients import (
    get_document_store as _get_document_store,
    get_embedder as _get_embedder,
    get_retriever as _get_retriever,
)
from clients.base import BaseDocumentStore, BaseEmbedder, BaseRetriever
from clients.factory import EmbedderType, VectorStoreType
from haystack import Document

logger = logging.getLogger(__name__)


# Convenience functions that wrap the client factory
def get_preferred_document_store() -> BaseDocumentStore:
    """Get document store using preferred configuration.

    Returns:
        BaseDocumentStore: The preferred document store instance.
    """
    return _get_document_store()


def get_preferred_retriever(top_k: Optional[int] = None) -> BaseRetriever:
    """Get retriever using preferred configuration.

    Args:
        top_k: The maximum number of documents to retrieve. If None, uses default.

    Returns:
        BaseRetriever: The preferred retriever instance.
    """
    return _get_retriever(top_k=top_k)


def get_preferred_embedder() -> BaseEmbedder:
    """Get embedder using preferred configuration.

    Returns:
        BaseEmbedder: The preferred embedder instance (Google embedder).
    """
    return _get_embedder(EmbedderType.GOOGLE)


# Async wrappers
async def get_document_store_async() -> BaseDocumentStore:
    """Async version of get_preferred_document_store.

    Returns:
        BaseDocumentStore: The preferred document store instance.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_preferred_document_store)


async def get_retriever_async(top_k: Optional[int] = None) -> BaseRetriever:
    """Async version of get_preferred_retriever.

    Args:
        top_k: The maximum number of documents to retrieve. If None, uses default.

    Returns:
        BaseRetriever: The preferred retriever instance.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_preferred_retriever, top_k)


async def get_embedder_async() -> BaseEmbedder:
    """Async version of get_preferred_embedder.

    Returns:
        BaseEmbedder: The preferred embedder instance.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_preferred_embedder)


# Document operations
def get_document_count_sync() -> int:
    """Get total document count synchronously.

    Returns:
        int: Total number of documents in the store, 0 if error occurs.
    """
    try:
        document_store = get_preferred_document_store()
        return document_store.count_documents()
    except Exception as e:
        logger.error(f"Error getting document count: {e}")
        return 0


async def get_document_count() -> int:
    """Get total document count asynchronously.

    Returns:
        int: Total number of documents in the store.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_document_count_sync)


def embed_documents_sync(documents: List[Document]) -> List[Document]:
    """Embed documents synchronously.

    Args:
        documents: List of documents to embed.

    Returns:
        List[Document]: Documents with embeddings added, original list if error occurs.
    """
    try:
        embedder = get_preferred_embedder()
        return embedder.embed_documents(documents)
    except Exception as e:
        logger.error(f"Error embedding documents: {e}")
        return documents


async def embed_documents(documents: List[Document]) -> List[Document]:
    """Embed documents asynchronously.

    Args:
        documents: List of documents to embed.

    Returns:
        List[Document]: Documents with embeddings added.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_documents_sync, documents)


def embed_query_sync(query: str) -> List[float]:
    """Embed query synchronously.

    Args:
        query: The query string to embed.

    Returns:
        List[float]: Query embedding vector, empty list if error occurs.
    """
    try:
        embedder = get_preferred_embedder()
        return embedder.embed_query(query)
    except Exception as e:
        logger.error(f"Error embedding query: {e}")
        return []


async def embed_query(query: str) -> List[float]:
    """Embed query asynchronously.

    Args:
        query: The query string to embed.

    Returns:
        List[float]: Query embedding vector.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_query_sync, query)


def retrieve_documents_sync(query: str, top_k: Optional[int] = None) -> List[Document]:
    """Retrieve documents synchronously.

    Args:
        query: The query string to search for.
        top_k: The maximum number of documents to retrieve. If None, uses default.

    Returns:
        List[Document]: Retrieved documents, empty list if error occurs.
    """
    try:
        # First embed the query
        query_embedding = embed_query_sync(query)
        if not query_embedding:
            return []

        # Then retrieve using the embedding
        retriever = get_preferred_retriever(top_k)
        return retriever.retrieve_with_embedding(query_embedding, top_k)
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []


async def retrieve_documents(query: str, top_k: Optional[int] = None) -> List[Document]:
    """Retrieve documents asynchronously.

    Args:
        query: The query string to search for.
        top_k: The maximum number of documents to retrieve. If None, uses default.

    Returns:
        List[Document]: Retrieved documents.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, retrieve_documents_sync, query, top_k)


# Store type specific functions for advanced usage
def get_chroma_document_store() -> BaseDocumentStore:
    """Get ChromaDB document store specifically.

    Returns:
        BaseDocumentStore: ChromaDB document store instance.
    """
    return _get_document_store(VectorStoreType.CHROMA)


def get_elasticsearch_document_store() -> BaseDocumentStore:
    """Get Elasticsearch document store specifically.

    Returns:
        BaseDocumentStore: Elasticsearch document store instance.
    """
    return _get_document_store(VectorStoreType.ELASTICSEARCH)


def get_chroma_retriever(top_k: Optional[int] = None) -> BaseRetriever:
    """Get ChromaDB retriever specifically.

    Args:
        top_k: The maximum number of documents to retrieve. If None, uses default.

    Returns:
        BaseRetriever: ChromaDB retriever instance.
    """
    return _get_retriever(VectorStoreType.CHROMA, top_k=top_k)


def get_elasticsearch_retriever(top_k: Optional[int] = None) -> BaseRetriever:
    """Get Elasticsearch retriever specifically.

    Args:
        top_k: The maximum number of documents to retrieve. If None, uses default.

    Returns:
        BaseRetriever: Elasticsearch retriever instance.
    """
    return _get_retriever(VectorStoreType.ELASTICSEARCH, top_k=top_k)
