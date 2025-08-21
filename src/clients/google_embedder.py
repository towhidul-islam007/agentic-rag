"""Google Generative AI embedder implementation"""

import logging

from typing import List

from haystack import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import get_settings
from clients.base import BaseEmbedder

logger = logging.getLogger(__name__)


class GoogleEmbedder(BaseEmbedder):
    """Google Generative AI embedder implementation"""

    def __init__(self, model: str = "models/embedding-001") -> None:
        self.settings = get_settings()

        if not self.settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for GoogleEmbedder")

        self.embedder = GoogleGenerativeAIEmbeddings(
            model=model, google_api_key=self.settings.google_api_key
        )
        self.model = model
        logger.info(f"GoogleEmbedder initialized with model: {model}")

    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """
        Embed documents using Google Generative AI embeddings

        Args:
            documents: List of documents to embed

        Returns:
            List of documents with embeddings added
        """
        if not documents:
            return documents

        try:
            # Extract text content from documents
            texts = [doc.content for doc in documents]

            # Generate embeddings
            embeddings = self.embedder.embed_documents(texts)

            # Add embeddings to documents
            for doc, embedding in zip(documents, embeddings, strict=True):
                doc.embedding = embedding

            logger.info(f"Successfully embedded {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string using Google Generative AI embeddings

        Args:
            query: Query string to embed

        Returns:
            Query embedding as list of floats
        """
        try:
            embedding = self.embedder.embed_query(query)
            logger.debug(f"Successfully embedded query: {query[:50]}...")
            return embedding

        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
