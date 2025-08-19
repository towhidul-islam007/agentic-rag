# Agentic Adaptive RAG Chatbot

An intelligent chatbot that combines document search and web search using LangGraph, Haystack, and Google's Gemini models. The system automatically decides whether to search your private documents, the web, or both based on your query.

## 🌟 Features

- **Agentic Query Routing**: Automatically decides the best search strategy for each query
- **Document Management**: Upload and process PDF, TXT, MD, and DOCX files
- **Vector Search**: Uses ChromaDB for efficient document retrieval
- **Web Search Integration**: Searches the web for current information
- **Adaptive RAG**: Combines multiple information sources intelligently
- **Production Ready**: Built with Haystack and LangGraph for scalability

## 🏗️ Architecture

The system uses:
- **LangGraph**: For building the agentic workflow
- **Haystack**: For document processing and RAG pipelines
- **ChromaDB**: For vector storage and similarity search
- **Sentence Transformers**: For generating embeddings
- **Google Gemini**: For language generation and query routing
- **Streamlit**: For the user interface

## 🚀 Quick Start

### Option 1: Streamlit Web Interface
```bash
poetry run streamlit run app.py
```

### Option 2: Python Script
```python
from src.rag_system import AgenticRAG

# Initialize system
rag = AgenticRAG()

# Add documents (optional)
rag.add_documents(["path/to/document.pdf"])

# Query the system
result = rag.query("What is artificial intelligence?")
print(result["response"])
```

### Option 3: Run Example
```bash
poetry run python example.py
```

## 🧪 Testing

Run comprehensive tests:
```bash
poetry run python test_agentic_rag.py
```

## 🚀 Full Setup

### Prerequisites
- Python 3.12+
- Poetry (install from https://python-poetry.org/docs/#installation)
- Google Cloud account with Vertex AI API enabled OR Google API key

### Installation

1. **Clone and setup the project:**
   ```bash
   git clone <your-repo>
   cd <your-repo>
   poetry install
   ```

2. **Run the setup script:**
   ```bash
   python setup.py
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your values:
   ```env
   GOOGLE_CLOUD_PROJECT=your-project-id
   VERTEX_AI_LOCATION=global
   GOOGLE_API_KEY=your-google-api-key  # Alternative to Vertex AI
   ```

4. **Set up Google Cloud (if using Vertex AI):**
   - Create a Google Cloud project
   - Enable the Vertex AI API
   - Set up Application Default Credentials:
     ```bash
     gcloud auth application-default login
     ```

### Running the App

```bash
poetry run streamlit run app.py
```

Or use the configured script:
```bash
poetry run start
```

## 📖 How to Use

1. **Upload Documents**: Use the file uploader to add PDF, TXT, MD, or DOCX files
2. **Ask Questions**: The system will automatically:
   - Route your query to the appropriate search method
   - Search your documents and/or the web
   - Generate a comprehensive response
3. **View Search Details**: Expand the search details to see how your query was processed

### Query Examples

- **Document-focused**: "What are the key findings in the research paper?"
- **Web-focused**: "What's the latest news about AI?"
- **Hybrid**: "How does the concept in my document relate to current industry trends?"

## 🔧 Configuration

### Environment Variables

- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud project ID
- `VERTEX_AI_LOCATION`: Vertex AI region (default: global)
- `GOOGLE_API_KEY`: Google API key (alternative to Vertex AI)
- `CHROMA_DB_PATH`: Path for ChromaDB storage (default: ./chroma_db)
- `UPLOAD_DIR`: Directory for uploaded files (default: ./uploads)

### Supported File Types

- **PDF**: Extracted using PyPDF
- **Text**: Plain text and Markdown files
- **Word**: DOCX files (converted to text)

## 🧠 How It Works

### 1. Query Routing
The system analyzes each query and decides:
- **Documents**: For domain-specific or technical questions
- **Web**: For current events or general knowledge
- **Both**: When combining sources would be beneficial

### 2. Document Processing
- Files are split into chunks with overlap
- Embeddings are generated using Sentence Transformers
- Stored in ChromaDB for efficient retrieval

### 3. Response Generation
- Relevant context is retrieved from chosen sources
- Gemini generates a comprehensive response
- Sources and search strategy are displayed

## 🛠️ Development

### Project Structure
```
├── src/                   # Core agentic RAG system
│   ├── __init__.py       # Package initialization
│   ├── state.py          # State definitions
│   ├── nodes.py          # Node implementations
│   ├── edges.py          # Edge logic and routing
│   ├── utils.py          # Utility functions
│   ├── graph.py          # Graph construction
│   └── rag_system.py     # Main RAG system
├── app.py                # Streamlit application
├── document_manager.py   # Document upload and processing
├── config.py             # Configuration settings
├── example.py            # Usage example
├── test_agentic_rag.py   # Comprehensive tests
├── pyproject.toml        # Poetry configuration
├── requirements.txt      # Dependencies
└── README.md             # This file
```

### Adding New Features

The system is designed to be extensible:
- Add new document types in `document_manager.py`
- Extend the LangGraph workflow in `rag_system.py`
- Add new search sources by creating new nodes

### Running Tests

```bash
poetry run pytest  # When tests are added
```

### Code Quality

```bash
poetry run ruff check    # Linting
poetry run mypy .       # Type checking
```

## 📊 Performance

- **Document Retrieval**: Sub-second similarity search with ChromaDB
- **Embedding Generation**: Optimized with Sentence Transformers
- **Response Generation**: Streaming responses from Gemini
- **Memory Usage**: Efficient document chunking and storage

## 🔒 Security

- Environment variables for sensitive data
- Local document storage (no data sent to external services except for generation)
- Configurable upload limits
- Input validation and sanitization

## 🆘 Troubleshooting

### Common Issues

1. **ChromaDB errors**: Delete the `chroma_db` directory and restart
2. **Authentication errors**: Check your Google Cloud credentials
3. **Memory issues**: Reduce document chunk size or upload smaller files
4. **Import errors**: Run `poetry install` to ensure all dependencies are installed

### Getting Help

- Check the logs in the Streamlit interface
- Review the search details for debugging
- Ensure all environment variables are set correctly
