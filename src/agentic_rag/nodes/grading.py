"""Document and response grading nodes"""

import logging

from agentic_rag.models import AnswerGrade, HallucinationGrade, RelevanceGrade
from agentic_rag.nodes.base import BaseNode
from agentic_rag.state import AgenticRAGState

logger = logging.getLogger(__name__)


class DocumentGrader(BaseNode):
    """Handles document relevance grading."""

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        """Initialize the document grader.

        Args:
            model: Model name to use. Defaults to 'gemini-2.5-flash'.
        """
        super().__init__(model)
        self.retrieval_grader = self.create_llm(
            temperature=0.0, structured_output=RelevanceGrade
        )

    async def grade_documents(self, state: AgenticRAGState) -> AgenticRAGState:
        """Grade document relevance to question.

        Args:
            state: Current state object.

        Returns:
            AgenticRAGState: The result.
        """
        question = state["question"]
        documents = state.get("documents", [])

        # Check if no documents were found in retrieval
        if not documents:
            state["filtered_documents"] = []
            state["document_relevance_scores"] = []
            state["relevance_grade"] = "not_relevant"
            logger.info("No documents to grade - setting relevance to not_relevant")
            return state

        filtered_docs = []
        relevance_scores = []

        for doc in documents:
            grading_prompt = f"""
            Assess the relevance of this retrieved document to the user question.

            Retrieved document:
            {doc}

            User question: {question}

            Determine if the document contains information relevant to answering
            the question.
            This doesn't need to be a stringent test - the goal is to filter out
            clearly irrelevant retrievals.
            """

            try:
                grade = await self.run_llm_call(self.retrieval_grader, grading_prompt)
                if grade.score == "yes":
                    filtered_docs.append(doc)
                    relevance_scores.append(1.0)
                else:
                    relevance_scores.append(0.0)
                logger.info(f"Document relevance: {grade.score} - {grade.reasoning}")

            except Exception as e:
                logger.error(f"Error grading document: {e}")
                # On error, keep the document to be safe
                filtered_docs.append(doc)
                relevance_scores.append(0.5)

        state["filtered_documents"] = filtered_docs
        state["document_relevance_scores"] = relevance_scores
        state["relevance_grade"] = "relevant" if filtered_docs else "not_relevant"

        logger.info(f"Filtered to {len(filtered_docs)} relevant documents")
        return state


class ResponseGrader(BaseNode):
    """Handles response quality grading."""

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        """Initialize the response grader.

        Args:
            model: Model name to use. Defaults to 'gemini-2.5-flash'.
        """
        super().__init__(model)
        self.hallucination_grader = self.create_llm(
            temperature=0.0, structured_output=HallucinationGrade
        )
        self.answer_grader = self.create_llm(
            temperature=0.0, structured_output=AnswerGrade
        )

    async def grade_generation_v_documents_and_question(
        self, state: AgenticRAGState
    ) -> AgenticRAGState:
        """Determines whether generation is grounded in document and answers question.

        Args:
            state: Current state object.

        Returns:
            AgenticRAGState: The result.
        """
        question = state["question"]
        documents = state.get("filtered_documents", [])
        generation = state.get("generation", "")

        if not documents:
            # Skip hallucination check if no documents
            state["hallucination_grade"] = "no"  # no hallucination when no docs
            logger.info("No documents provided - skipping hallucination check")
        else:
            doc_context = "\n".join(documents[:3])

            # Grade for hallucinations
            hallucination_prompt = f"""
            Assess whether this LLM generation is grounded in and supported by
            the provided facts.

            Set of facts:
            {doc_context}

            LLM generation: {generation}

            Determine if the answer is supported by the facts or contains
            unsupported claims.
            """

            try:
                hallucination_grade = await self.run_llm_call(
                    self.hallucination_grader, hallucination_prompt
                )
                state["hallucination_grade"] = hallucination_grade.score
                logger.info(
                    f"Hallucination grade: {hallucination_grade.score} - "
                    f"{hallucination_grade.reasoning}"
                )

            except Exception as e:
                logger.error(f"Error in hallucination grading: {e}")
                state["hallucination_grade"] = "no"  # Default to no hallucination

        # Grade for answer quality
        answer_prompt = f"""
        Assess whether this answer effectively addresses and resolves the
        user's question.

        User question: {question}
        LLM generation: {generation}

        Determine if the answer is useful and relevant to the question.
        """

        try:
            answer_grade = await self.run_llm_call(self.answer_grader, answer_prompt)
            state["answer_grade"] = answer_grade.score
            logger.info(
                f"Answer grade: {answer_grade.score} - {answer_grade.reasoning}"
            )

        except Exception as e:
            logger.error(f"Error in answer grading: {e}")
            state["answer_grade"] = "useful"  # Default to useful

        return state
