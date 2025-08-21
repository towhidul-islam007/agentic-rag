"""Document retrieval node"""

import asyncio
import logging

from typing import List

from haystack import Document

from src.agentic_rag.nodes.base import BaseNode
from src.agentic_rag.state import AgenticRAGState
from src.utils import get_preferred_embedder, get_preferred_retriever

logger = logging.getLogger(__name__)


class DocumentRetriever(BaseNode):
    """Handles document retrieval from vector store"""

    def __init__(self, gemini_model: str = "gemini-2.5-flash") -> None:
        super().__init__(gemini_model)
        # Initialize retriever and embedder
        self.retriever = get_preferred_retriever()
        self.text_embedder = get_preferred_embedder()

    async def retrieve(self, state: AgenticRAGState) -> AgenticRAGState:
        """Retrieve documents"""
        question = state["question"]

        try:
            # Embed the query synchronously (embedder is now sync)
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                None, self.text_embedder.embed_query, question
            )

            # Retrieve documents
            def run_retriever() -> List[Document]:
                return self.retriever.retrieve_with_embedding(query_embedding)

            documents = await loop.run_in_executor(None, run_retriever)
            # Convert Document objects to strings for state
            doc_strings = [doc.content for doc in documents]
            state["documents"] = doc_strings
            logger.info(f"Retrieved {len(documents)} documents")

            # If no documents found, set a flag to indicate we should try web search
            if not documents:
                state["no_documents_found"] = True
                logger.info("No documents found in retrieval")

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            state["documents"] = []
            state["no_documents_found"] = True

        return state
