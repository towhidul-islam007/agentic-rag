"""Web search node"""

import asyncio
import logging

import requests

from bs4 import BeautifulSoup

from config import WEB_SEARCH_TIMEOUT
from src.agentic_rag.models import WebSearchQuery
from src.agentic_rag.nodes.base import BaseNode
from src.agentic_rag.state import AgenticRAGState

logger = logging.getLogger(__name__)


class WebSearcher(BaseNode):
    """Handles web search functionality"""

    def __init__(self, gemini_model: str = "gemini-2.5-flash") -> None:
        super().__init__(gemini_model)
        self.web_search_optimizer = self.create_llm(
            temperature=0.0, structured_output=WebSearchQuery
        )

    async def web_search_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Web search based on the question"""
        question = state["question"]

        # Create a focused search query
        search_query_prompt = f"""
        Create an optimized web search query for this question: "{question}"

        Make it specific and likely to return good search results.
        Consider what terms would appear in relevant web pages.
        """

        try:
            web_query = await self.run_llm_call(
                self.web_search_optimizer, search_query_prompt
            )
            search_query = web_query.search_query
            state["web_search_query"] = search_query
            logger.info(f"Web search query: {search_query} - {web_query.reasoning}")
        except Exception as e:
            logger.error(f"Error optimizing web search query: {e}")
            search_query = question
            state["web_search_query"] = search_query

        try:
            # Perform web search
            search_url = f"https://duckduckgo.com/html/?q={search_query}"

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(
                    search_url,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=WEB_SEARCH_TIMEOUT,
                ),
            )

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                results = []

                # Extract search results
                for result in soup.find_all("a", class_="result__a")[:3]:
                    title = result.get_text(strip=True)
                    if title:
                        results.append(title)

                state["web_results"] = results
                logger.info(f"Found {len(results)} web search results")
            else:
                logger.warning(f"Web search failed with status: {response.status_code}")
                state["web_results"] = []

        except Exception as e:
            logger.error(f"Error in web search: {e}")
            state["web_results"] = []

        return state
