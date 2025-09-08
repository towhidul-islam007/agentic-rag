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


def create_graph(
    model: str = "gemini-2.5-flash", fast_mode: bool = True, use_documents: bool = True
) -> Any:
    """Create and compile the RAG workflow graph.

    Args:
        model: The model name to use for LLM operations.
        fast_mode: If True, creates a simplified pipeline for faster responses.
        use_documents: If True, includes document retrieval;
                      if False, direct LLM response.

    Returns:
        Any: The compiled workflow graph.
    """
    nodes = RAGNodes(model=model, fast_mode=fast_mode)

    # Create the graph
    workflow = StateGraph(AgenticRAGState)

    if not use_documents:
        # Direct LLM mode - just generate without retrieval
        workflow.add_node("generate", nodes.generate)
        workflow.set_entry_point("generate")
        workflow.add_edge("generate", END)

    elif fast_mode:
        # Simplified pipeline for speed: just retrieve and generate
        workflow.add_node("retrieve", nodes.retrieve)
        workflow.add_node("generate", nodes.generate)

        # Simple linear flow
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

    else:
        # Original complex pipeline for thorough processing
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

        # Build complex graph
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
                "not supported": END,
                "useful": END,
                "not useful": END,
            },
        )

    # Compile
    return workflow.compile()
