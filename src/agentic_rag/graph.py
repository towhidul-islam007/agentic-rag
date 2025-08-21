"""Main graph construction for the Agentic RAG system"""

from typing import Any

from agentic_rag.edges import (
    decide_to_generate,
    grade_generation_v_documents_and_question,
    route_question,
)
from agentic_rag.nodes import RAGNodes
from agentic_rag.state import AgenticRAGState
from langgraph.graph import END, StateGraph


def create_graph(model: str = "gemini-2.5-flash") -> Any:
    """Create and compile the RAG workflow graph.

    Args:
        model: The model name to use for LLM operations.

    Returns:
        Any: The result.
    """
    nodes = RAGNodes(model=model)

    # Create the graph
    workflow = StateGraph(AgenticRAGState)

    # Define the nodes
    workflow.add_node("analyze_query", nodes.analyze_query)
    workflow.add_node("route_question", nodes.route_question)
    workflow.add_node("websearch", nodes.web_search_node)
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("generate", nodes.generate)
    workflow.add_node("transform_query", nodes.transform_query)
    workflow.add_node(
        "grade_generation_v_documents_and_question",
        nodes.grade_generation_v_documents_and_question,
    )

    # Build graph
    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "route_question")
    workflow.add_conditional_edges(
        "route_question",
        route_question,
        {"websearch": "websearch", "vectorstore": "retrieve"},
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"websearch": "websearch", "generate": "generate"},
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_edge("generate", "grade_generation_v_documents_and_question")
    workflow.add_conditional_edges(
        "grade_generation_v_documents_and_question",
        grade_generation_v_documents_and_question,
        {
            "not supported": END,  # Just end to avoid loops
            "useful": END,
            "not useful": END,  # Just end to avoid loops
        },
    )

    # Compile
    return workflow.compile()
