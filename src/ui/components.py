"""UI components for the Agentic RAG Streamlit application."""

from pathlib import Path

import streamlit as st

from document_management import DocumentManager
from rag_system import AgenticRAG

from config import get_settings
from src.utils import run_async

DEFAULT_MODEL = "gemini-2.5-flash"


def validate_configuration() -> None:
    """Validate application configuration and display errors if found."""
    settings = get_settings()
    config_errors = settings.validate_config()
    if config_errors:
        st.error("Configuration errors found:")
        for error in config_errors:
            st.error(f"• {error}")
        st.stop()


def configure_page() -> None:
    """Configure Streamlit page settings and title."""
    st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖", layout="wide")


def initialize_session_state() -> None:
    """Initialize Streamlit session state with default values."""
    settings = get_settings()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "show_document_management" not in st.session_state:
        st.session_state.show_document_management = False

    if "fast_mode" not in st.session_state:
        st.session_state.fast_mode = True

    if "use_documents" not in st.session_state:
        st.session_state.use_documents = False

    if "selected_model" not in st.session_state:
        # Get Google models and set default
        google_models = settings.available_models.get("google", {})
        if google_models:
            # Use default if available, otherwise use first available
            if settings.default_llm_model in google_models:
                st.session_state.selected_model = settings.default_llm_model
            else:
                st.session_state.selected_model = next(iter(google_models.keys()))
        else:
            st.session_state.selected_model = settings.default_llm_model


def init_rag_system(
    model_name: str, _fast_mode: bool = False, _use_documents: bool = True
) -> AgenticRAG | None:
    """Initialize RAG system with caching.

    Args:
        model_name: Name of the LLM model to use.
        _fast_mode: Whether to optimize for speed over thoroughness.
        _use_documents: Whether to include document retrieval or use direct LLM.

    Returns:
        AgenticRAG instance or None if initialization fails.
    """
    try:
        return AgenticRAG(
            model=model_name, fast_mode=_fast_mode, use_documents=_use_documents
        )
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None


def initialize_systems() -> None:
    """Initialize RAG system and document manager."""
    fast_mode = st.session_state.get("fast_mode", True)
    use_documents = st.session_state.get("use_documents", False)
    current_config = f"{st.session_state.selected_model}_{fast_mode}_{use_documents}"

    if "rag_system" not in st.session_state:
        with st.spinner("Initializing RAG System..."):
            st.session_state.rag_system = init_rag_system(
                st.session_state.selected_model,
                _fast_mode=fast_mode,
                _use_documents=use_documents,
            )
            st.session_state.document_manager = DocumentManager()
            st.session_state.current_config = current_config

    # Reinitialize if model, fast mode, or use_documents changed
    if st.session_state.get("current_config") != current_config:
        st.session_state.current_config = current_config
        # Reinitialize RAG system with new configuration
        with st.spinner("Updating configuration..."):
            st.session_state.rag_system = init_rag_system(
                st.session_state.selected_model,
                _fast_mode=fast_mode,
                _use_documents=use_documents,
            )
        # Reset chat session when configuration changes
        if "chat_history" in st.session_state:
            del st.session_state.chat_history


def display_chat_message(message: dict) -> None:
    """Display a single chat message with metadata if available.

    Args:
        message: Dictionary containing message role, content, and optional metadata.
    """
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show RAG metadata if available
        if message["role"] == "assistant" and "metadata" in message:
            metadata = message["metadata"]
            with st.expander("🔍 Search Details", expanded=False):
                st.caption(
                    f"**Route Decision:** {metadata.get('route_decision', 'N/A')}"
                )
                st.caption(f"**Documents Used:** {metadata.get('num_documents', 0)}")
                st.caption(f"**Web Results:** {metadata.get('num_web_results', 0)}")


def handle_user_input(prompt: str) -> None:
    """Handle user input and generate response.

    Args:
        prompt: User's input message.
    """
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using RAG system
    if st.session_state.rag_system:
        with (
            st.chat_message("assistant"),
            st.spinner("Analyzing query and searching for information..."),
        ):
            try:
                # Use RAG system to generate response asynchronously
                rag_result = run_async(st.session_state.rag_system.query(prompt))

                response_text = rag_result["response"]
                st.markdown(response_text)

                # Show search details
                with st.expander("🔍 Search Details", expanded=False):
                    st.caption(f"**Route Decision:** {rag_result['route_decision']}")
                    st.caption(f"**Documents Used:** {rag_result['num_documents']}")
                    st.caption(f"**Web Results:** {rag_result['num_web_results']}")

                    if rag_result["context"]:
                        st.caption("**Context Used:**")
                        st.text_area("Context", rag_result["context"], height=100)

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
                st.rerun()

            except Exception as e:
                error_msg = f"Error: {e!s}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
    else:
        st.error("RAG system not initialized. Please check your configuration.")


def render_chat_interface() -> None:
    """Render the main chat interface with controls."""
    # Show either chat interface or document management
    if st.session_state.show_document_management:
        render_document_management_content()
        return

    # Create a padded container using columns for better visual appearance
    col1, col2, col3 = st.columns([1, 6, 1])  # Add padding with side columns

    with col2:
        # Check if there are any messages to determine layout
        has_messages = len(st.session_state.messages) > 0

        # If no messages, add spacing to center the interface
        if not has_messages:
            st.markdown("<br>" * 3, unsafe_allow_html=True)

        # Display chat history
        for message in st.session_state.messages:
            display_chat_message(message)

        # Handle chat input at the very end
        prompt = st.chat_input("What would you like to know?")
        if prompt:
            handle_user_input(prompt)


def render_chat_controls() -> None:
    """Render the chat control toggles."""
    # Create two columns for the controls
    ctrl_col1, ctrl_col2 = st.columns(2)

    with ctrl_col1:
        # Fast response toggle
        fast_mode = st.toggle(
            "⚡ Fast Response",
            value=st.session_state.get("fast_mode", True),
            help="Skip analysis steps for faster responses",
            key="fast_mode_toggle",
        )
        if st.session_state.get("fast_mode") != fast_mode:
            st.session_state.fast_mode = fast_mode
            # Force reinitialization by clearing the RAG system
            st.session_state.rag_system = None
            st.session_state.pop("current_config", None)
            initialize_systems()
            st.rerun()

    with ctrl_col2:
        # Document search toggle
        use_documents = st.toggle(
            "📚 Search Documents",
            value=st.session_state.get("use_documents", False),
            help="Search uploaded documents vs direct LLM response",
            key="use_documents_toggle",
        )
        if st.session_state.get("use_documents") != use_documents:
            st.session_state.use_documents = use_documents
            # Force reinitialization by clearing the RAG system
            st.session_state.rag_system = None
            st.session_state.pop("current_config", None)
            initialize_systems()
            st.rerun()


def handle_document_upload(uploaded_files: list) -> None:
    """Handle document upload and processing.

    Args:
        uploaded_files: List of uploaded files from Streamlit file uploader.
    """
    with st.spinner("Processing documents..."):
        try:
            # Save uploaded files
            saved_paths = run_async(
                st.session_state.document_manager.save_uploaded_files(uploaded_files)
            )

            if saved_paths:
                # Process DOCX files
                processed_paths = run_async(
                    st.session_state.document_manager.process_docx_files(saved_paths)
                )

                # Add to RAG system asynchronously
                success = run_async(
                    st.session_state.rag_system.add_documents(processed_paths)
                )

                if success:
                    st.success(
                        f"Successfully processed {len(processed_paths)} documents!"
                    )
                else:
                    st.error("Error processing documents")
            else:
                st.warning("No valid files to process")

        except Exception as e:
            st.error(f"Error processing documents: {e}")


def render_document_management_content() -> None:
    """Render the document management interface content."""
    st.header("📚 Document Management")

    if not (st.session_state.document_manager and st.session_state.rag_system):
        st.error("Document management not available")
        return

    # Document upload
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        help="Upload PDF, TXT, MD, or DOCX files to add to your knowledge base",
    )

    if uploaded_files and st.button("Process Documents"):
        handle_document_upload(uploaded_files)

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


def render_model_selector() -> None:
    """Render the model selection interface in the sidebar."""
    st.header("⚙️ Configuration")

    settings = get_settings()
    # Get Google models since Azure is removed
    google_models = settings.available_models.get("google", {})

    if not google_models:
        st.error("No models available")
        return

    # Model selector
    model_keys = list(google_models.keys())
    current_model = st.session_state.get("selected_model", settings.default_llm_model)

    # Ensure current model is valid
    if current_model not in google_models:
        current_model = model_keys[0] if model_keys else settings.default_llm_model

    try:
        current_index = model_keys.index(current_model)
    except ValueError:
        current_index = 0

    selected_model = st.selectbox(
        "Choose LLM Model:",
        options=model_keys,
        format_func=lambda x: google_models[x]["name"],
        index=current_index,
        key="model_selector",
    )

    # Update selected model if changed
    if selected_model != st.session_state.get("selected_model"):
        st.session_state.selected_model = selected_model
        st.rerun()

    # Display model description
    if selected_model in google_models:
        st.info(
            f"**{google_models[selected_model]['name']}**\n\n"
            f"{google_models[selected_model]['description']}"
        )

    # Chat controls in sidebar
    st.divider()
    st.subheader("🎛️ Chat Controls")

    # Fast response toggle
    fast_mode = st.toggle(
        "⚡ Fast Response",
        value=st.session_state.get("fast_mode", True),
        help="Skip analysis steps for faster responses",
        key="sidebar_fast_mode_toggle",
    )
    if st.session_state.get("fast_mode") != fast_mode:
        st.session_state.fast_mode = fast_mode
        # Force reinitialization by clearing the RAG system
        st.session_state.rag_system = None
        st.session_state.pop("current_config", None)
        initialize_systems()
        st.rerun()

    # Document search toggle
    use_documents = st.toggle(
        "📚 Search Documents",
        value=st.session_state.get("use_documents", False),
        help="Search uploaded documents vs direct LLM response",
        key="sidebar_use_documents_toggle",
    )
    if st.session_state.get("use_documents") != use_documents:
        st.session_state.use_documents = use_documents
        # Force reinitialization by clearing the RAG system
        st.session_state.rag_system = None
        st.session_state.pop("current_config", None)
        initialize_systems()
        st.rerun()


def render_system_info() -> None:
    """Render RAG system information in the sidebar."""
    st.divider()
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


def render_sidebar() -> None:
    """Render the complete sidebar interface."""
    with st.sidebar:
        # Document management toggle button
        if st.button(
            "📚 Document Management"
            if not st.session_state.show_document_management
            else "💬 Back to Chat",
            key="toggle_doc_mgmt",
            use_container_width=True,
        ):
            st.session_state.show_document_management = (
                not st.session_state.show_document_management
            )
            st.rerun()

        st.divider()

        # Model selection
        render_model_selector()

        st.divider()

        # Clear chat history button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
