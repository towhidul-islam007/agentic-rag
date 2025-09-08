"""Agentic RAG Chatbot - Streamlit Application

This module provides a Streamlit-based web interface for the Agentic RAG system,
allowing users to interact with documents and web search through a chat interface.
"""

import logging

import nest_asyncio

from dotenv import load_dotenv

from src.ui import (
    configure_page,
    initialize_session_state,
    initialize_systems,
    render_chat_interface,
    render_sidebar,
    validate_configuration,
)

# Apply nest_asyncio to allow nested event loops in Streamlit
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    """Main application entry point."""
    # Configuration and initialization
    validate_configuration()
    configure_page()
    initialize_session_state()
    initialize_systems()

    # Main chat interface (full width)
    render_chat_interface()

    # Sidebar
    render_sidebar()


if __name__ == "__main__":
    main()
