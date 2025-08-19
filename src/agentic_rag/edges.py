"""Edge logic for the Agentic RAG system"""

import logging

from src.agentic_rag.state import AgenticRAGState
from src.utils import get_document_count_sync

logger = logging.getLogger(__name__)


def route_question(state: AgenticRAGState) -> str:
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    print("---ROUTE QUESTION---")
    route_decision = state.get("route_decision", "vectorstore")

    # Check if we have any documents in the system first
    doc_count = get_document_count_sync()

    if doc_count == 0:
        print(f"---NO DOCUMENTS IN SYSTEM ({doc_count}), FORCE WEB SEARCH---")
        return "websearch"

    if route_decision == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    if route_decision == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    return "websearch"  # Default fallback


def decide_to_generate(state: AgenticRAGState) -> str:
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    print("---ASSESS GRADED DOCUMENTS---")
    relevance_grade = state.get("relevance_grade", "relevant")
    filtered_documents = state.get("filtered_documents", [])
    web_results = state.get("web_results", [])
    no_documents_found = state.get("no_documents_found", False)

    # If no documents were found in retrieval, go to web search
    if no_documents_found:
        print("---DECISION: NO DOCUMENTS IN DATABASE, ROUTE TO WEB SEARCH---")
        return "websearch"

    if relevance_grade == "not_relevant" or not filtered_documents:
        # Check if we already have web results
        if web_results:
            print("---DECISION: NO RELEVANT DOCS BUT HAVE WEB RESULTS, GENERATE---")
            return "generate"
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    # We have relevant documents, so generate answer
    print("---DECISION: GENERATE---")
    return "generate"


def grade_generation_v_documents_and_question(state: AgenticRAGState) -> str:  # noqa
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    print("---CHECK HALLUCINATIONS---")
    hallucination_grade = state.get("hallucination_grade", "no")
    answer_grade = state.get("answer_grade", "useful")
    loop_count = state.get("loop_count", 0)
    max_loops = state.get("max_loops", 3)
    filtered_documents = state.get("filtered_documents", [])

    # Check for max retries
    if loop_count >= max_loops:
        print("---DECISION: MAX RETRIES REACHED---")
        return "useful"

    # If no documents, skip hallucination check and just check answer quality
    if not filtered_documents:
        print("---NO DOCUMENTS: CHECKING ANSWER QUALITY ONLY---")
        if answer_grade == "useful":
            print("---DECISION: ANSWER IS USEFUL---")
            return "useful"
        print("---DECISION: ANSWER NOT USEFUL, BUT MAX RETRIES TO AVOID LOOP---")
        # Don't retry when there are no documents, just accept the answer
        return "useful"

    # With documents, check hallucination first
    if hallucination_grade == "no":  # "no" means not hallucinated (grounded)
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        if answer_grade == "useful":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        # If we've tried multiple times, just accept the answer to avoid infinite loops
        if loop_count >= 1:
            print("---DECISION: ACCEPTING ANSWER TO AVOID INFINITE LOOP---")
            return "useful"
        state["loop_count"] = loop_count + 1
        return "not useful"
    print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
    # If we've tried multiple times, just accept the answer to avoid infinite loops
    if loop_count >= 1:
        print("---DECISION: TOO MANY RETRIES, ACCEPTING ANSWER---")
        return "useful"
    state["loop_count"] = loop_count + 1
    return "not supported"
