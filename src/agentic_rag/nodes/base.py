"""Base node class with common utilities"""

import asyncio
import concurrent.futures
import logging

from typing import Any, Awaitable, Callable

from langchain_google_genai import ChatGoogleGenerativeAI

from config import get_settings

# Get settings instance
settings = get_settings()

logger = logging.getLogger(__name__)

# Thread pool executor for LLM calls
_llm_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="llm_"
)


def run_llm_sync_safe(
    llm_func: Callable[[str], Awaitable[Any]], prompt: str
) -> Callable[[], Any]:
    """Run LLM async function synchronously in a separate thread.

    Args:
        llm_func: The llm_func parameter.
        prompt: The prompt parameter.

    Returns:
        Callable[[], Any]: The result.
    """

    def _run_sync() -> Any:
        """Run Sync.

        Returns:
            Any: The result of the operation.
        """
        try:
            # Import nest_asyncio to patch asyncio if needed
            try:
                import nest_asyncio

                nest_asyncio.apply()
            except ImportError:
                pass

            # Use asyncio.run which creates and manages its own event loop
            return asyncio.run(llm_func(prompt))
        except Exception as e:
            logger.error(f"Error in LLM sync execution: {e}")
            raise

    return _run_sync


class BaseNode:
    """Base class for all nodes with common LLM initialization"""

    def __init__(self, gemini_model: str = "gemini-2.5-flash") -> None:
        """Initialize the base node with LLM configuration.

        Args:
            gemini_model: Model name or configuration. Defaults to 'gemini-2.5-flash'.
        """
        # Initialize LLM config
        self.llm_config = (
            {"model": gemini_model, "google_api_key": settings.google_api_key}
            if settings.google_api_key
            else {"model": gemini_model}
        )

    def create_llm(
        self, temperature: float = 0.0, structured_output: Any = None
    ) -> Any:
        """Create an LLM instance with the given configuration.

        Args:
            temperature: The temperature setting for the LLM (0.0 to 1.0).
            structured_output: Optional structured output schema for the LLM.

        Returns:
            Any: Configured LLM instance.
        """
        llm = ChatGoogleGenerativeAI(temperature=temperature, **self.llm_config)
        if structured_output:
            return llm.with_structured_output(structured_output)
        return llm

    async def run_llm_call(self, llm: Any, prompt: str) -> Any:
        """Run an LLM call safely in a separate thread.

        Args:
            llm: The LLM instance to call.
            prompt: The prompt string to send to the LLM.

        Returns:
            Any: The LLM response.
        """
        loop = asyncio.get_event_loop()
        llm_task = run_llm_sync_safe(llm.ainvoke, prompt)
        return await loop.run_in_executor(_llm_executor, llm_task)
