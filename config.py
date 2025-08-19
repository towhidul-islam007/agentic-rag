"""Configuration settings for the RAG system"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google Cloud / Vertex AI Configuration
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
VERTEX_AI_LOCATION = os.getenv("VERTEX_AI_LOCATION", "global")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# RAG System Configuration
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"
)
DEFAULT_GEMINI_MODEL = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-2.5-flash")

# Document Processing Configuration
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))

# Search Configuration
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
WEB_SEARCH_TIMEOUT = int(os.getenv("WEB_SEARCH_TIMEOUT", "10"))

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def validate_config() -> list[str]:
    """Validate configuration settings"""
    errors = []

    if not GOOGLE_CLOUD_PROJECT and not GOOGLE_API_KEY:
        errors.append("Either GOOGLE_CLOUD_PROJECT or GOOGLE_API_KEY must be set")

    # Create necessary directories
    Path(CHROMA_DB_PATH).mkdir(exist_ok=True)
    Path(UPLOAD_DIR).mkdir(exist_ok=True)

    return errors


def get_config_summary() -> dict:
    """Get a summary of current configuration"""
    return {
        "google_cloud_project": GOOGLE_CLOUD_PROJECT,
        "vertex_ai_location": VERTEX_AI_LOCATION,
        "has_google_api_key": bool(GOOGLE_API_KEY),
        "chroma_db_path": CHROMA_DB_PATH,
        "upload_dir": UPLOAD_DIR,
        "embedding_model": EMBEDDING_MODEL,
        "default_gemini_model": DEFAULT_GEMINI_MODEL,
        "max_chunk_size": MAX_CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "default_top_k": DEFAULT_TOP_K,
    }
