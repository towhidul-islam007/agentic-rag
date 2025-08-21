"""Haystack component wrappers for the modular client system"""

import logging

from typing import Any, Dict, List

from haystack import Document, component

from clients.base import BaseDocumentStore, BaseEmbedder

logger = logging.getLogger(__name__)


@component
class EmbedderWrapper:
    """Haystack component wrapper for BaseEmbedder implementations"""

    def __init__(self, embedder: BaseEmbedder) -> None:
        self.embedder = embedder

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Embed documents using the wrapped embedder

        Args:
            documents: List of documents to embed

        Returns:
            Dict with embedded documents
        """
        try:
            embedded_docs = self.embedder.embed_documents(documents)
            return {"documents": embedded_docs}
        except Exception as e:
            logger.error(f"Error in EmbedderWrapper: {e}")
            return {"documents": documents}


@component
class DocumentStoreWriter:
    """Haystack component wrapper for BaseDocumentStore implementations"""

    def __init__(self, document_store: BaseDocumentStore) -> None:
        self.document_store = document_store

    @component.output_types(documents_written=int)
    def run(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Write documents to the wrapped document store

        Args:
            documents: List of documents to write

        Returns:
            Dict with number of documents written
        """
        try:
            self.document_store.write_documents(documents)
            return {"documents_written": len(documents)}
        except Exception as e:
            logger.error(f"Error in DocumentStoreWriter: {e}")
            return {"documents_written": 0}
