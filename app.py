import os

import streamlit as st

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

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
        "description": "Gemini 2.0 Flash model optimized for cost efficiency and low latency",
    },
}


def init_genai_client() -> genai.Client | None:
    """Initialize GenAI Client."""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("VERTEX_AI_LOCATION", "us-central1")

    if not project_id:
        st.error("Please set GOOGLE_CLOUD_PROJECT in your .env file")
        return None

    return genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )


# Streamlit app configuration
st.set_page_config(page_title="Vertex AI Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Gemini AI Chatbot")
st.markdown("Chat with Google's Gemini models")

# Initialize selected model in session state
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemini-2.5-flash"

# Initialize GenAI client
if "genai_client" not in st.session_state:
    with st.spinner("Initializing GenAI Client..."):
        st.session_state.genai_client = init_genai_client()

# Track current model for reinitialization
if st.session_state.get("current_model") != st.session_state.selected_model:
    st.session_state.current_model = st.session_state.selected_model
    # Reset chat session when model changes
    if "chat_history" in st.session_state:
        del st.session_state.chat_history

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    if st.session_state.genai_client:
        with st.chat_message("assistant"):  # noqa
            with st.spinner("Thinking..."):
                try:
                    # Initialize chat history if not exists
                    if "chat_history" not in st.session_state:
                        st.session_state.chat_history = []

                    # Build conversation history
                    contents = []
                    for msg in st.session_state.messages:
                        role = "user" if msg["role"] == "user" else "model"
                        contents.append(
                            types.Content(
                                role=role,
                                parts=[types.Part(text=msg["content"])],
                            ),
                        )

                    # Add current user message
                    contents.append(
                        types.Content(role="user", parts=[types.Part(text=prompt)]),
                    )

                    # Generate response
                    response_text = ""
                    for (
                        chunk
                    ) in st.session_state.genai_client.models.generate_content_stream(
                        model=st.session_state.selected_model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            temperature=0.7,
                            top_p=0.95,
                            max_output_tokens=8192,
                        ),
                    ):
                        if chunk.text:
                            response_text += chunk.text

                    st.markdown(response_text)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text},
                    )

                except Exception as e:  # noqa
                    error_msg = f"Error: {e!s}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg},
                    )
    else:
        st.error("GenAI client not initialized. Please check your configuration.")

# Sidebar with model selection and configuration
with st.sidebar:
    st.header("Model Selection")

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
        f"**{GEMINI_MODELS[st.session_state.selected_model]['name']}**\n\n{GEMINI_MODELS[st.session_state.selected_model]['description']}",
    )

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        # Reset chat history to start fresh
        if "chat_history" in st.session_state:
            del st.session_state.chat_history
        st.rerun()
