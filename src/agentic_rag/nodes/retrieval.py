"""Document retrieval node"""

import asyncio
import logging

from typing import List

from agentic_rag.nodes.base import BaseNode
from agentic_rag.state import AgenticRAGState
from haystack import Document
from utils import get_preferred_embedder, get_preferred_retriever

logger = logging.getLogger(__name__)


class DocumentRetriever(BaseNode):
    """Handles document retrieval from vector store."""

    def __init__(self, model: str = "gemini-2.5-flash", fast_mode: bool = True) -> None:
        """Initialize the document retriever.

        Args:
            model: The model name to use for LLM operations.
            fast_mode: If True, creates a simplified pipeline for faster responses.
        """
        super().__init__(model)
        # Initialize retriever and embedder
        self.retriever = get_preferred_retriever()
        self.text_embedder = get_preferred_embedder()
        self.fast_mode = fast_mode

    async def retrieve(self, state: AgenticRAGState) -> AgenticRAGState:
        """Retrieve documents based on the question in the state.

        Args:
            state: The current state containing the question.

        Returns:
            AgenticRAGState: Updated state with retrieved documents.
        """
        question = state["question"]

        try:
            # Embed the query synchronously (embedder is now sync)
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                None, self.text_embedder.embed_query, question
            )

            # Retrieve documents
            def run_retriever() -> List[Document]:
                """Run Retriever.

                Returns:
                    List[Document]: List of results.
                """
                return self.retriever.retrieve_with_embedding(query_embedding)

            documents = await loop.run_in_executor(None, run_retriever)
            # Convert Document objects to strings for state
            doc_strings = [doc.content for doc in documents]
            state["documents"] = doc_strings
            logger.info(f"Retrieved {len(documents)} documents")

            if self.fast_mode:
                # In fast mode, set filtered_documents to documents directly
                state["filtered_documents"] = doc_strings

            # If no documents found, set a flag to indicate we should try web search
            if not documents:
                state["no_documents_found"] = True
                logger.info("No documents found in retrieval")

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            state["documents"] = []
            state["no_documents_found"] = True

        return state
