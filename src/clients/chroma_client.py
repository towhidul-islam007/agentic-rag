"""ChromaDB implementation for document storage and retrieval"""

import logging

from pathlib import Path
from typing import List, Optional

from haystack import Document
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from config import get_settings
from clients.base import BaseDocumentStore, BaseRetriever

logger = logging.getLogger(__name__)


class ChromaDocumentStoreWrapper(BaseDocumentStore):
    """ChromaDB document store wrapper"""

    def __init__(self, collection_name: str = "documents") -> None:
        self.settings = get_settings()

        # Create ChromaDB document store
        chroma_path = Path(self.settings.chroma_db_path)
        chroma_path.mkdir(exist_ok=True)

        try:
            self.document_store = ChromaDocumentStore(
                collection_name=collection_name, persist_path=str(chroma_path)
            )
            logger.info(f"ChromaDB initialized with collection: {collection_name}")
        except Exception as e:
            logger.warning(f"Error initializing ChromaDB: {e}")
            # Try with a different collection name
            self.document_store = ChromaDocumentStore(
                collection_name=f"{collection_name}_fallback",
                persist_path=str(chroma_path),
            )
            logger.info("ChromaDB initialized with fallback collection")

    def write_documents(self, documents: List[Document]) -> None:
        """Write documents to ChromaDB"""
        try:
            self.document_store.write_documents(documents)
            logger.info(f"Successfully wrote {len(documents)} documents to ChromaDB")
        except Exception as e:
            logger.error(f"Error writing documents to ChromaDB: {e}")
            raise

    def count_documents(self) -> int:
        """Count documents in ChromaDB"""
        try:
            return self.document_store.count_documents()
        except Exception as e:
            logger.error(f"Error counting documents in ChromaDB: {e}")
            return 0

    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from ChromaDB"""
        try:
            self.document_store.delete_documents(document_ids)
            logger.info(f"Successfully deleted {len(document_ids)} documents")
        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {e}")
            raise

    def get_all_documents(self) -> List[Document]:
        """Get all documents from ChromaDB"""
        try:
            return self.document_store.get_all_documents()
        except Exception as e:
            logger.error(f"Error getting documents from ChromaDB: {e}")
            return []


class ChromaRetriever(BaseRetriever):
    """ChromaDB retriever implementation"""

    def __init__(
        self, document_store: ChromaDocumentStoreWrapper, top_k: Optional[int] = None
    ) -> None:
        self.settings = get_settings()
        self.document_store = document_store
        self.top_k = top_k or self.settings.default_top_k

        self.retriever = ChromaEmbeddingRetriever(
            document_store=document_store.document_store, top_k=self.top_k
        )
        logger.info(f"ChromaRetriever initialized with top_k: {self.top_k}")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """Retrieve documents based on query"""
        # Note: This would require query embedding first
        # For now, this is a placeholder - actual implementation would need
        # the embedder to convert query to embedding first
        raise NotImplementedError(
            "Direct query retrieval not implemented. Use retrieve_with_embedding."
        )

    def retrieve_with_embedding(
        self, query_embedding: List[float], top_k: Optional[int] = None
    ) -> List[Document]:
        """Retrieve documents based on query embedding"""
        try:
            k = top_k or self.top_k
            result = self.retriever.run(query_embedding=query_embedding, top_k=k)
            documents = result.get("documents", [])
            logger.info(f"Retrieved {len(documents)} documents from ChromaDB")
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents from ChromaDB: {e}")
            return []
