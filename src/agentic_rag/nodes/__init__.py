"""Node implementations for the Agentic RAG system"""

from agentic_rag.nodes.analysis import QueryAnalyzer
from agentic_rag.nodes.generation import ResponseGenerator
from agentic_rag.nodes.grading import DocumentGrader, ResponseGrader
from agentic_rag.nodes.retrieval import DocumentRetriever
from agentic_rag.nodes.search import WebSearcher
from agentic_rag.state import AgenticRAGState


class RAGNodes:
    """Collection of nodes for the RAG workflow"""

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        # Initialize all node components
        self.query_analyzer = QueryAnalyzer(model)
        self.document_retriever = DocumentRetriever(model)
        self.document_grader = DocumentGrader(model)
        self.response_generator = ResponseGenerator(model)
        self.response_grader = ResponseGrader(model)
        self.web_searcher = WebSearcher(model)

    async def analyze_query(self, state: AgenticRAGState) -> AgenticRAGState:
        """Analyze the query to understand intent and complexity"""
        return await self.query_analyzer.analyze_query(state)

    async def route_question(self, state: AgenticRAGState) -> AgenticRAGState:
        """Route question to appropriate retrieval method"""
        return await self.query_analyzer.route_question(state)

    async def retrieve(self, state: AgenticRAGState) -> AgenticRAGState:
        """Retrieve documents"""
        return await self.document_retriever.retrieve(state)

    async def grade_documents(self, state: AgenticRAGState) -> AgenticRAGState:
        """Grade document relevance to question"""
        return await self.document_grader.grade_documents(state)

    async def generate(self, state: AgenticRAGState) -> AgenticRAGState:
        """Generate answer"""
        return await self.response_generator.generate(state)

    async def transform_query(self, state: AgenticRAGState) -> AgenticRAGState:
        """Transform the query to produce a better question"""
        return await self.query_analyzer.transform_query(state)

    async def web_search_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Web search based on the question"""
        return await self.web_searcher.web_search_node(state)

    async def grade_generation_v_documents_and_question(
        self, state: AgenticRAGState
    ) -> AgenticRAGState:
        """Determines whether the generation is grounded in the document and answers question"""
        return await self.response_grader.grade_generation_v_documents_and_question(
            state
        )
