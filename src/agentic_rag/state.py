"""State definitions for the Agentic RAG system"""

from typing import Any, Dict, List, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]
    messages: Annotated[List[BaseMessage], add_messages]


class AgenticRAGState(TypedDict):
    """Enhanced state for Agentic RAG workflow"""

    # Input
    question: str
    messages: Annotated[List[BaseMessage], add_messages]

    # Query analysis
    query_analysis: Dict[str, Any]
    route_decision: Literal["vectorstore", "websearch", "both"]

    # Document retrieval
    documents: List[str]
    document_relevance_scores: List[float]
    filtered_documents: List[str]
    no_documents_found: bool

    # Web search
    web_results: List[str]
    web_search_query: str

    # Generation
    context: str
    generation: str

    # Grading
    relevance_grade: Literal["relevant", "not_relevant"]
    hallucination_grade: Literal["yes", "no"]
    answer_grade: Literal["useful", "not useful"]

    # Loop control
    loop_count: int
    max_loops: int
