"""Query analysis and routing nodes"""

import logging

from agentic_rag.models import QueryAnalysis, QueryTransformation, RouteDecision
from agentic_rag.nodes.base import BaseNode
from agentic_rag.state import AgenticRAGState

logger = logging.getLogger(__name__)


class QueryAnalyzer(BaseNode):
    """Handles query analysis and routing logic."""

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        """Initialize the query analyzer.

        Args:
            model: Model name to use. Defaults to 'gemini-2.5-flash'.
        """
        super().__init__(model)
        self.query_analyzer = self.create_llm(
            temperature=0.0, structured_output=QueryAnalysis
        )
        self.route_decider = self.create_llm(
            temperature=0.0, structured_output=RouteDecision
        )
        self.query_transformer = self.create_llm(
            temperature=0.0, structured_output=QueryTransformation
        )

    async def analyze_query(self, state: AgenticRAGState) -> AgenticRAGState:
        """Analyze the query to understand intent and complexity.

        Args:
            state: Current state object.

        Returns:
            AgenticRAGState: The result.
        """
        question = state["question"]

        analysis_prompt = f"""
        Analyze the following question:

        Question: "{question}"

        Provide analysis covering:
        1. The main topic or domain of the question
        2. Complexity level (simple, medium, or complex)
        3. Information type needed (factual, analytical, or creative)
        4. Most likely source for answering (documents, web, or both)
        """

        try:
            analysis = await self.run_llm_call(self.query_analyzer, analysis_prompt)

            # Convert Pydantic model to dict for state storage
            state["query_analysis"] = {
                "topic": analysis.topic,
                "complexity": analysis.complexity,
                "info_type": analysis.info_type,
                "likely_source": analysis.likely_source,
            }
            logger.info(f"Query analysis: {state['query_analysis']}")

        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            state["query_analysis"] = {
                "topic": "general",
                "complexity": "medium",
                "info_type": "factual",
                "likely_source": "documents",
            }

        return state

    async def route_question(self, state: AgenticRAGState) -> AgenticRAGState:
        """Route question to appropriate retrieval method.

        Args:
            state: Current state object.

        Returns:
            AgenticRAGState: The result.
        """
        question = state["question"]

        routing_prompt = f"""
        Decide whether to route this question to vectorstore or websearch.

        Question: "{question}"

        Use "vectorstore" for questions about documents, technical topics, or
        specific knowledge.
        Use "websearch" for current events, news, or general questions.

        Provide your routing decision with reasoning.
        """

        try:
            route_decision = await self.run_llm_call(self.route_decider, routing_prompt)
            state["route_decision"] = route_decision.route
            logger.info(
                f"Route decision: {route_decision.route} - {route_decision.reasoning}"
            )

        except Exception as e:
            logger.error(f"Error in routing: {e}")
            state["route_decision"] = "vectorstore"

        return state

    async def transform_query(self, state: AgenticRAGState) -> AgenticRAGState:
        """Transform the query to produce a better question.

        Args:
            state: Current state object.

        Returns:
            AgenticRAGState: The result.
        """
        question = state["question"]

        transform_prompt = f"""
        Transform this question to be better optimized for document retrieval.
        Consider the underlying semantic intent and meaning.

        Original question: {question}

        Provide an improved version that would better match relevant documents.
        """

        try:
            transformation = await self.run_llm_call(
                self.query_transformer, transform_prompt
            )
            state["question"] = transformation.transformed_query
            logger.info(
                f"Transformed query: {transformation.transformed_query} - "
                f"{transformation.reasoning}"
            )

        except Exception as e:
            logger.error(f"Error transforming query: {e}")

        return state
