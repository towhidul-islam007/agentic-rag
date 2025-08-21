"""Elasticsearch implementation for document storage and retrieval"""

import logging

from typing import List, Optional

from haystack import Document
from haystack_integrations.components.retrievers.elasticsearch import (
    ElasticsearchEmbeddingRetriever,
)
from haystack_integrations.document_stores.elasticsearch import (
    ElasticsearchDocumentStore,
)

from config import get_settings
from clients.base import BaseDocumentStore, BaseRetriever

logger = logging.getLogger(__name__)


class ElasticsearchDocumentStoreWrapper(BaseDocumentStore):
    """Elasticsearch document store wrapper"""

    def __init__(self, index: Optional[str] = None) -> None:
        self.settings = get_settings()

        if (
            not self.settings.elasticsearch_url
            or not self.settings.elasticsearch_api_key
        ):
            raise ValueError("Elasticsearch URL and API key are required")

        self.index = index or self.settings.elasticsearch_index

        try:
            # Create Elasticsearch document store
            hosts = [self.settings.elasticsearch_url]

            # Prepare connection parameters
            connection_params = {
                "headers": {
                    "Authorization": f"ApiKey {self.settings.elasticsearch_api_key}"
                },
                "verify_certs": True,
            }

            # Add CA certs if provided
            if self.settings.elasticsearch_ca_certs:
                connection_params["ca_certs"] = self.settings.elasticsearch_ca_certs

            self.document_store = ElasticsearchDocumentStore(
                hosts=hosts, index=self.index, **connection_params
            )

            logger.info(
                f"Elasticsearch document store initialized with index: {self.index}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch document store: {e}")
            raise

    def write_documents(self, documents: List[Document]) -> None:
        """Write documents to Elasticsearch"""
        try:
            self.document_store.write_documents(documents)
            logger.info(
                f"Successfully wrote {len(documents)} documents to Elasticsearch"
            )
        except Exception as e:
            logger.error(f"Error writing documents to Elasticsearch: {e}")
            raise

    def count_documents(self) -> int:
        """Count documents in Elasticsearch"""
        try:
            return self.document_store.count_documents()
        except Exception as e:
            logger.error(f"Error counting documents in Elasticsearch: {e}")
            return 0

    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from Elasticsearch"""
        try:
            self.document_store.delete_documents(document_ids)
            logger.info(f"Successfully deleted {len(document_ids)} documents")
        except Exception as e:
            logger.error(f"Error deleting documents from Elasticsearch: {e}")
            raise

    def get_all_documents(self) -> List[Document]:
        """Get all documents from Elasticsearch"""
        try:
            return self.document_store.get_all_documents()
        except Exception as e:
            logger.error(f"Error getting documents from Elasticsearch: {e}")
            return []


class ElasticsearchRetriever(BaseRetriever):
    """Elasticsearch retriever implementation"""

    def __init__(
        self,
        document_store: ElasticsearchDocumentStoreWrapper,
        top_k: Optional[int] = None,
    ) -> None:
        self.settings = get_settings()
        self.document_store = document_store
        self.top_k = top_k or self.settings.default_top_k

        self.retriever = ElasticsearchEmbeddingRetriever(
            document_store=document_store.document_store, top_k=self.top_k
        )
        logger.info(f"ElasticsearchRetriever initialized with top_k: {self.top_k}")

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
            logger.info(f"Retrieved {len(documents)} documents from Elasticsearch")
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents from Elasticsearch: {e}")
            return []
