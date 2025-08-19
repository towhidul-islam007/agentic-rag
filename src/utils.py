"""Utility functions for the Agentic RAG system"""

import asyncio
import logging
from pathlib import Path

from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import get_settings
from src.embedders import GoogleDocumentEmbedder, GoogleTextEmbedder

# Get settings instance
settings = get_settings()

logger = logging.getLogger(__name__)

# Global instances
_document_store = None
_retriever = None
_text_embedder = None
_google_embedder = None


def get_document_store():
    """Get or create ChromaDB document store instance"""
    global _document_store

    if _document_store is None:
        if not settings.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required for ChromaDB with Google embeddings"
            )

        # Create ChromaDB document store
        chroma_path = Path(settings.chroma_db_path)
        chroma_path.mkdir(exist_ok=True)

        try:
            # Try to create/connect to the document store
            _document_store = ChromaDocumentStore(
                collection_name="documents", persist_path=str(chroma_path)
            )
            logger.info("ChromaDB document store initialized")
        except Exception as e:
            logger.warning(f"Error initializing ChromaDB: {e}")
            # If there's an embedding dimension mismatch, create a new collection
            _document_store = ChromaDocumentStore(
                collection_name="documents_google", persist_path=str(chroma_path)
            )
            logger.info(
                "ChromaDB document store initialized with new collection for "
                "Google embeddings"
            )

    return _document_store


async def get_document_store_async():
    """Async version of get_document_store"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_document_store)


def get_retriever():
    """Get or create ChromaDB retriever instance"""
    global _retriever

    if _retriever is None:
        document_store = get_document_store()
        _retriever = ChromaEmbeddingRetriever(
            document_store=document_store, top_k=settings.default_top_k
        )
        logger.info("ChromaDB retriever initialized")

    return _retriever


async def get_retriever_async():
    """Async version of get_retriever"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_retriever)


def get_embedder():
    """Get or create Google text embedder instance"""
    global _text_embedder

    if _text_embedder is None:
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for Google text embeddings")

        _text_embedder = GoogleTextEmbedder()
        logger.info("Google text embedder initialized")

    return _text_embedder


async def get_embedder_async():
    """Async version of get_embedder"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_embedder)


def get_google_embedder():
    """Get or create Google Generative AI embedder instance"""
    global _google_embedder

    if _google_embedder is None and settings.google_api_key:
        try:
            _google_embedder = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=settings.google_api_key
            )
            logger.info("Google Generative AI embedder initialized")
        except Exception as e:
            logger.error(f"Error initializing Google embedder: {e}")
            _google_embedder = None

    return _google_embedder


async def get_google_embedder_async():
    """Async version of get_google_embedder"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_google_embedder)


def get_document_embedder():
    """Get or create Google document embedder instance"""
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY is required for Google document embeddings")

    return GoogleDocumentEmbedder()


async def get_document_embedder_async():
    """Async version of get_document_embedder"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_document_embedder)


async def save_document_store() -> bool:
    """ChromaDB automatically persists data, so this is a no-op"""
    try:
        # ChromaDB automatically persists data to disk
        logger.info("ChromaDB automatically persists data")
        return True
    except Exception as e:
        logger.error(f"Error with document store: {e}")
        return False


def save_document_store_sync() -> bool:
    """Synchronous version of save_document_store"""
    try:
        # ChromaDB automatically persists data to disk
        logger.info("ChromaDB automatically persists data")
        return True
    except Exception as e:
        logger.error(f"Error with document store: {e}")
        return False


async def get_document_count() -> int:
    """Get the number of documents in the store"""
    try:
        document_store = await get_document_store_async()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, document_store.count_documents)
    except Exception:
        return 0


def get_document_count_sync() -> int:
    """Synchronous version of get_document_count"""
    try:
        document_store = get_document_store()
        return document_store.count_documents()
    except Exception:
        return 0
