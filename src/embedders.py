"""Embedders for the Agentic RAG system"""

import logging

from typing import Any, Dict, List

from haystack import Document, component
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import get_settings

# Get settings instance
settings = get_settings()

logger = logging.getLogger(__name__)


@component
class GoogleDocumentEmbedder:
    """Document embedder using Google Generative AI embeddings"""

    def __init__(self, model: str = "models/embedding-001") -> None:
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for GoogleDocumentEmbedder")

        self.embedder = GoogleGenerativeAIEmbeddings(
            model=model, google_api_key=settings.google_api_key
        )
        self.model = model
        logger.info(f"GoogleDocumentEmbedder initialized with model: {model}")

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Embed documents using Google Generative AI embeddings

        Args:
            documents: List of documents to embed

        Returns:
            Dict with embedded documents
        """
        if not documents:
            return {"documents": []}

        try:
            # Extract text content from documents
            texts = [doc.content for doc in documents]

            # Generate embeddings synchronously
            embeddings = self.embedder.embed_documents(texts)

            # Add embeddings to documents
            embedded_docs = []
            for doc, embedding in zip(documents, embeddings, strict=False):
                embedded_doc = Document(
                    content=doc.content, meta=doc.meta, embedding=embedding
                )
                embedded_docs.append(embedded_doc)

            logger.info(
                f"Embedded {len(embedded_docs)} documents using Google embeddings"
            )
            return {"documents": embedded_docs}

        except Exception as e:
            logger.error(f"Error embedding documents with Google embeddings: {e}")
            # Fallback: return documents without embeddings
            return {"documents": documents}


@component
class GoogleTextEmbedder:
    """Text embedder using Google Generative AI embeddings"""

    def __init__(self, model: str = "models/embedding-001") -> None:
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for GoogleTextEmbedder")

        self.embedder = GoogleGenerativeAIEmbeddings(
            model=model, google_api_key=settings.google_api_key
        )
        self.model = model
        logger.info(f"GoogleTextEmbedder initialized with model: {model}")

    @component.output_types(embedding=List[float])
    def run(self, text: str) -> Dict[str, Any]:
        """
        Embed text using Google Generative AI embeddings

        Args:
            text: Text to embed

        Returns:
            Dict with embedding
        """
        if not text:
            return {"embedding": []}

        try:
            # Generate embedding synchronously
            embedding = self.embedder.embed_query(text)

            logger.debug(f"Generated embedding for text: {text[:50]}...")
            return {"embedding": embedding}

        except Exception as e:
            logger.error(f"Error embedding text with Google embeddings: {e}")
            return {"embedding": []}
