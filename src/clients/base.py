"""Base abstract classes for vector stores and embedders"""

from abc import ABC, abstractmethod
from typing import List, Optional

from haystack import Document


class BaseEmbedder(ABC):
    """Abstract base class for document embedders"""

    @abstractmethod
    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """
        Embed documents and return documents with embeddings

        Args:
            documents: List of documents to embed

        Returns:
            List of documents with embeddings added
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string

        Args:
            query: Query string to embed

        Returns:
            Query embedding as list of floats
        """
        ...


class BaseDocumentStore(ABC):
    """Abstract base class for document stores"""

    @abstractmethod
    def write_documents(self, documents: List[Document]) -> None:
        """
        Write documents to the store

        Args:
            documents: List of documents to store
        """
        ...

    @abstractmethod
    def count_documents(self) -> int:
        """
        Count total documents in the store

        Returns:
            Number of documents
        """
        ...

    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents by IDs

        Args:
            document_ids: List of document IDs to delete
        """
        ...

    @abstractmethod
    def get_all_documents(self) -> List[Document]:
        """
        Get all documents from the store

        Returns:
            List of all documents
        """
        ...


class BaseRetriever(ABC):
    """Abstract base class for retrievers"""

    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Retrieve documents based on query

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents
        """
        ...

    @abstractmethod
    def retrieve_with_embedding(
        self, query_embedding: List[float], top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Retrieve documents based on query embedding

        Args:
            query_embedding: Query embedding vector
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents
        """
        ...
