import asyncio
import logging

from pathlib import Path
from typing import Any, Coroutine

import nest_asyncio
import streamlit as st

from document_management import DocumentManager
from dotenv import load_dotenv
from rag_system import AgenticRAG

from config import get_settings

# Apply nest_asyncio to allow nested event loops in Streamlit
nest_asyncio.apply()

# Load environment variables
load_dotenv()


def run_async(coro: Coroutine) -> Any:
    """Helper function to run async functions in Streamlit"""
    try:
        # With nest_asyncio, we can use asyncio.run even in running loops
        return asyncio.run(coro)
    except RuntimeError:
        # Fallback to get_event_loop if needed
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


# Available Gemini models
GEMINI_MODELS = {
    "gemini-2.5-pro": {
        "name": "Gemini 2.5 Pro",
        "description": "Our most advanced reasoning model to date",
    },
    "gemini-2.5-flash": {
        "name": "Gemini 2.5 Flash",
        "description": "Best price-performance, offering well-rounded capabilities",
    },
    "gemini-2.5-flash-lite": {
        "name": "Gemini 2.5 Flash-Lite",
        "description": "Most cost effective model that supports high throughput tasks",
    },
    "gemini-2.0-flash-exp": {
        "name": "Gemini 2.0 Flash",
        "description": "Newest multimodal model, with next generation features",
    },
    "gemini-2.0-flash-lite": {
        "name": "Gemini 2.0 Flash-Lite",
        "description": (
            "Gemini 2.0 Flash model optimized for cost efficiency and low latency"
        ),
    },
}


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get settings instance
settings = get_settings()

# Validate configuration
config_errors = settings.validate_config()
if config_errors:
    st.error("Configuration errors found:")
    for error in config_errors:
        st.error(f"• {error}")
    st.stop()

# Streamlit app configuration
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Agentic Adaptive RAG Chatbot")
st.markdown(
    "Chat with an intelligent system that can search your documents and the web"
)

# Initialize selected model in session state
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemini-2.5-flash"


# Initialize RAG system
@st.cache_resource
def init_rag_system(model_name: str) -> AgenticRAG | None:
    """Initialize RAG system with caching"""
    try:
        return AgenticRAG(gemini_model=model_name)
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None


if "rag_system" not in st.session_state:
    with st.spinner("Initializing RAG System..."):
        st.session_state.rag_system = init_rag_system(st.session_state.selected_model)
        st.session_state.document_manager = DocumentManager()

# Track current model for reinitialization
if st.session_state.get("current_model") != st.session_state.selected_model:
    st.session_state.current_model = st.session_state.selected_model
    # Reset chat session when model changes
    if "chat_history" in st.session_state:
        del st.session_state.chat_history
    # Reinitialize RAG system with new model
    st.session_state.rag_system = init_rag_system(st.session_state.selected_model)

# Create main layout
col1, col2 = st.columns([2, 1])

with col1:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show RAG metadata if available
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                with st.expander("🔍 Search Details", expanded=False):
                    st.write(
                        f"**Route Decision:** {metadata.get('route_decision', 'N/A')}"
                    )
                    st.write(f"**Documents Used:** {metadata.get('num_documents', 0)}")
                    st.write(f"**Web Results:** {metadata.get('num_web_results', 0)}")

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response using RAG system
        if st.session_state.rag_system:
            with st.chat_message("assistant"):  # noqa
                with st.spinner("Analyzing query and searching for information..."):
                    try:
                        # Use RAG system to generate response asynchronously
                        rag_result = run_async(
                            st.session_state.rag_system.query(prompt)
                        )

                        response_text = rag_result["response"]
                        st.markdown(response_text)

                        # Show search details
                        with st.expander("🔍 Search Details", expanded=False):
                            st.write(
                                f"**Route Decision:** {rag_result['route_decision']}"
                            )
                            st.write(
                                f"**Documents Used:** {rag_result['num_documents']}"
                            )
                            st.write(
                                f"**Web Results:** {rag_result['num_web_results']}"
                            )

                            if rag_result["context"]:
                                st.write("**Context Used:**")
                                st.text_area(
                                    "Context", rag_result["context"], height=200
                                )

                        # Add to chat history with metadata
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response_text,
                                "metadata": {
                                    "route_decision": rag_result["route_decision"],
                                    "num_documents": rag_result["num_documents"],
                                    "num_web_results": rag_result["num_web_results"],
                                },
                            }
                        )

                    except Exception as e:
                        error_msg = f"Error: {e!s}"
                        st.error(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )
        else:
            st.error("RAG system not initialized. Please check your configuration.")

with col2:
    st.header("📚 Document Management")

    if st.session_state.document_manager and st.session_state.rag_system:
        # Document upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            help="Upload PDF, TXT, MD, or DOCX files to add to your knowledge base",
        )

        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    # Save uploaded files
                    saved_paths = run_async(
                        st.session_state.document_manager.save_uploaded_files(
                            uploaded_files
                        )
                    )

                    if saved_paths:
                        # Process DOCX files
                        processed_paths = run_async(
                            st.session_state.document_manager.process_docx_files(
                                saved_paths
                            )
                        )

                        # Add to RAG system asynchronously
                        success = run_async(
                            st.session_state.rag_system.add_documents(processed_paths)
                        )

                        if success:
                            st.success(
                                f"Successfully processed {len(processed_paths)} "
                                f"documents!"
                            )
                        else:
                            st.error("Error processing documents")
                    else:
                        st.warning("No valid files to process")

                except Exception as e:
                    st.error(f"Error processing documents: {e}")

        # Document statistics
        st.subheader("📊 Knowledge Base Stats")
        doc_count = st.session_state.rag_system.get_document_count()
        st.metric("Documents in Knowledge Base", doc_count)

        # Uploaded files management
        uploaded_files_list = run_async(
            st.session_state.document_manager.get_uploaded_files()
        )
        if uploaded_files_list:
            st.subheader("📁 Uploaded Files")
            for file_path in uploaded_files_list:
                file_name = Path(file_path).name
                col_file, col_delete = st.columns([3, 1])
                with col_file:
                    st.text(file_name)
                with col_delete:
                    if st.button("🗑️", key=f"delete_{file_name}") and run_async(
                        st.session_state.document_manager.delete_file(file_path)
                    ):
                        st.rerun()

        # Clear all documents
        if st.button("🗑️ Clear All Documents", type="secondary") and run_async(
            st.session_state.document_manager.clear_all_files()
        ):
            st.success("All documents cleared!")
            st.rerun()

    else:
        st.error("Document management not available")

# Sidebar with configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Model selector
    selected_model = st.selectbox(
        "Choose Gemini Model:",
        options=list(GEMINI_MODELS.keys()),
        format_func=lambda x: GEMINI_MODELS[x]["name"],
        index=list(GEMINI_MODELS.keys()).index(st.session_state.selected_model),
        key="model_selector",
    )

    # Update selected model if changed
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.rerun()

    # Display model description
    st.info(
        f"**{GEMINI_MODELS[st.session_state.selected_model]['name']}**\n\n{GEMINI_MODELS[st.session_state.selected_model]['description']}"
    )

    st.divider()

    # RAG System Info
    st.header("🧠 RAG System")

    if st.session_state.rag_system:
        st.success("✅ RAG System Active")

        # System capabilities
        st.subheader("🎯 Capabilities")
        st.write("• Document search and retrieval")
        st.write("• Web search integration")
        st.write("• Intelligent query routing")
        st.write("• Context-aware responses")

        # Search strategy info
        st.subheader("🔍 Search Strategy")
        st.write("The system automatically decides whether to:")
        st.write("• Search your documents")
        st.write("• Search the web")
        st.write("• Use both sources")

    else:
        st.error("❌ RAG System Unavailable")

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        # Reset chat history to start fresh
        if "chat_history" in st.session_state:
            del st.session_state.chat_history
        st.rerun()

    st.divider()

    # Help section
    st.header("❓ How to Use")
    st.write("1. **Upload documents** using the file uploader")
    st.write(
        "2. **Ask questions** - the system will intelligently search your "
        "documents and/or the web"
    )
    st.write(
        "3. **View search details** in the expandable sections to see how "
        "your query was processed"
    )

    st.subheader("💡 Tips")
    st.write("• Upload PDFs, text files, or Word documents")
    st.write("• Ask specific questions about your documents")
    st.write("• Ask general questions for web search")
    st.write("• The system will combine sources when helpful")
