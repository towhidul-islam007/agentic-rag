"""Pydantic models for structured LLM responses"""

from typing import Literal

from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    """Model for query analysis response"""

    topic: str = Field(description="The main topic or domain of the question")
    complexity: Literal["simple", "medium", "complex"] = Field(
        description="The complexity level of the question"
    )
    info_type: Literal["factual", "analytical", "creative"] = Field(
        description="The type of information needed"
    )
    likely_source: Literal["documents", "web", "both"] = Field(
        description="The most likely source for answering this question"
    )


class RouteDecision(BaseModel):
    """Model for routing decision"""

    route: Literal["vectorstore", "websearch"] = Field(
        description="Where to route the query"
    )
    reasoning: str = Field(description="Brief explanation for the routing decision")


class RelevanceGrade(BaseModel):
    """Model for document relevance grading"""

    score: Literal["yes", "no"] = Field(
        description="Whether the document is relevant to the question"
    )
    reasoning: str = Field(description="Brief explanation for the relevance score")


class HallucinationGrade(BaseModel):
    """Model for hallucination detection"""

    score: Literal["yes", "no"] = Field(
        description=(
            "Whether the answer is grounded in the provided documents "
            "(yes = grounded, no = hallucinated)"
        )
    )
    reasoning: str = Field(
        description="Brief explanation for the hallucination assessment"
    )


class AnswerGrade(BaseModel):
    """Model for answer quality assessment"""

    score: Literal["useful", "not useful"] = Field(
        description="Whether the answer addresses the user's question"
    )
    reasoning: str = Field(description="Brief explanation for the answer assessment")


class QueryTransformation(BaseModel):
    """Model for query transformation"""

    transformed_query: str = Field(
        description="The improved version of the original query"
    )
    reasoning: str = Field(
        description="Brief explanation for why the query was transformed this way"
    )


class WebSearchQuery(BaseModel):
    """Model for web search query optimization"""

    search_query: str = Field(description="Optimized search query for web search")
    reasoning: str = Field(
        description="Brief explanation for the search query optimization"
    )
