"""Response generation node"""

import logging

from agentic_rag.nodes.base import BaseNode
from agentic_rag.state import AgenticRAGState

logger = logging.getLogger(__name__)


class ResponseGenerator(BaseNode):
    """Handles response generation."""

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        """Initialize the response generator.

        Args:
            model: Model name to use. Defaults to 'gemini-2.5-flash'.
        """
        super().__init__(model)
        self.generator = self.create_llm(
            temperature=0.3
        )  # Higher temperature for generation

    async def generate(self, state: AgenticRAGState) -> AgenticRAGState:
        """Generate answer.

        Args:
            state: Current state object.

        Returns:
            AgenticRAGState: The result.
        """
        question = state["question"]
        documents = state.get("filtered_documents", [])
        web_results = state.get("web_results", [])

        # Build context
        context_parts = []

        if documents:
            doc_context = "\n\n".join(documents[:3])
            context_parts.append(f"Document Context:\n{doc_context}")

        if web_results:
            web_context = "\n".join(web_results)
            context_parts.append(f"Web Search Results:\n{web_context}")

        context = "\n\n---\n\n".join(context_parts)
        state["context"] = context

        # Generate response
        if context.strip():
            generation_prompt = f"""
            You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.

            Question: {question}
            Context: {context}
            Answer:
            """
        else:
            generation_prompt = f"""
            You are an assistant for question-answering tasks.
            Answer the following question using your knowledge.

            Question: {question}
            Answer:
            """

        try:
            response = await self.run_llm_call(self.generator, generation_prompt)
            # Handle both string responses and objects with content attribute
            if hasattr(response, "content"):
                state["generation"] = response.content
            else:
                state["generation"] = str(response)
            logger.info("Generated response successfully")

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state["generation"] = (
                "I apologize, but I encountered an error while generating a response."
            )

        return state
