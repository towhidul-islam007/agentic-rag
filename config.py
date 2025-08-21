"""Configuration settings for the RAG system using Pydantic Settings"""

from functools import lru_cache
from pathlib import Path
from typing import Optional, Set

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation and type hints"""

    # Google Cloud / Vertex AI Configuration
    google_cloud_project: Optional[str] = Field(
        default=None, description="Google Cloud Project ID for Vertex AI"
    )
    vertex_ai_location: str = Field(
        default="global", description="Vertex AI location/region"
    )
    google_api_key: Optional[str] = Field(
        default=None, description="Google API key for Generative AI"
    )

    # Azure OpenAI Configuration
    azure_openai_api_key: Optional[str] = Field(
        default=None, description="Azure OpenAI API key"
    )
    azure_openai_endpoint: Optional[str] = Field(
        default=None, description="Azure OpenAI endpoint URL"
    )
    azure_openai_api_version: str = Field(
        default="2024-02-01", description="Azure OpenAI API version"
    )
    azure_openai_deployment_name: str = Field(
        default="gpt-4", description="Azure OpenAI deployment/model name"
    )

    # Elasticsearch Configuration
    elasticsearch_url: Optional[str] = Field(
        default=None, description="Elasticsearch cluster URL"
    )
    elasticsearch_api_key: Optional[str] = Field(
        default=None, description="Elasticsearch API key"
    )
    elasticsearch_index: str = Field(
        default="documents", description="Elasticsearch index name for documents"
    )
    elasticsearch_ca_certs: Optional[str] = Field(
        default=None, description="Path to CA certificates for Elasticsearch"
    )

    # RAG System Configuration
    chroma_db_path: Path = Field(
        default=Path("./chroma_db"), description="Path to ChromaDB storage directory"
    )
    upload_dir: Path = Field(
        default=Path("./uploads"), description="Directory for uploaded documents"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="Embedding model identifier",
    )
    default_llm_model: str = Field(
        default="gemini-2.5-flash", description="Default LLM model to use"
    )
    llm_provider: str = Field(
        default="google", description="LLM provider to use: 'google' or 'azure'"
    )

    # Document Processing Configuration
    max_chunk_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum chunk size for document splitting",
    )
    chunk_overlap: int = Field(
        default=200, ge=0, le=1000, description="Overlap between document chunks"
    )
    max_upload_size_mb: int = Field(
        default=200, ge=1, le=1000, description="Maximum file upload size in MB"
    )

    # Search Configuration
    default_top_k: int = Field(
        default=5, ge=1, le=100, description="Default number of documents to retrieve"
    )
    web_search_timeout: int = Field(
        default=10, ge=1, le=60, description="Web search timeout in seconds"
    )

    # Supported file extensions
    supported_extensions: Set[str] = Field(
        default={".pdf", ".txt", ".md", ".docx"},
        description="Supported file extensions for upload",
    )

    # Available LLM models configuration
    available_models: dict = Field(
        default={
            "gemini-2.5-pro": {
                "name": "Gemini 2.5 Pro",
                "description": "Our most advanced reasoning model to date",
            },
            "gemini-2.5-flash": {
                "name": "Gemini 2.5 Flash",
                "description": (
                    "Best price-performance, offering well-rounded capabilities"
                ),
            },
            "gemini-2.5-flash-lite": {
                "name": "Gemini 2.5 Flash-Lite",
                "description": (
                    "Most cost effective model that supports high throughput tasks"
                ),
            },
            "gemini-2.0-flash-exp": {
                "name": "Gemini 2.0 Flash",
                "description": "Newest multimodal model, with next generation features",
            },
            "gemini-2.0-flash-lite": {
                "name": "Gemini 2.0 Flash-Lite",
                "description": (
                    "Gemini 2.0 Flash model optimized for cost efficiency "
                    "and low latency"
                ),
            },
        },
        description="Available LLM models and their descriptions",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "validate_assignment": True,
        "extra": "allow",
    }

    @field_validator("chroma_db_path", "upload_dir")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Ensure directories exist"""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @model_validator(mode="after")
    def validate_chunk_overlap_and_credentials(self) -> "Settings":
        """Validate chunk overlap and Google credentials"""
        # Validate chunk overlap
        if self.chunk_overlap >= self.max_chunk_size:
            raise ValueError("chunk_overlap must be less than max_chunk_size")

        # Validate Google credentials
        if (
            self.llm_provider == "google"
            and not self.google_cloud_project
            and not self.google_api_key
        ):
            msg = (
                "Either google_cloud_project or google_api_key must be set "
                "when using Google LLM provider"
            )
            raise ValueError(msg)

        # Validate Azure OpenAI credentials
        if self.llm_provider == "azure" and (
            not self.azure_openai_api_key or not self.azure_openai_endpoint
        ):
            msg = (
                "azure_openai_api_key and azure_openai_endpoint must be set "
                "when using Azure LLM provider"
            )
            raise ValueError(msg)

        # Validate Elasticsearch credentials if URL is provided
        if self.elasticsearch_url and not self.elasticsearch_api_key:
            raise ValueError(
                "elasticsearch_api_key must be provided when elasticsearch_url is set"
            )

        return self

    def validate_config(self) -> list[str]:
        """Additional validation with detailed error messages"""
        errors = []

        if self.llm_provider == "google":
            if not self.google_cloud_project and not self.google_api_key:
                errors.append(
                    "Either GOOGLE_CLOUD_PROJECT or GOOGLE_API_KEY must be set"
                )
        elif self.llm_provider == "azure":
            if not self.azure_openai_api_key:
                errors.append(
                    "AZURE_OPENAI_API_KEY must be set when using Azure provider"
                )
            if not self.azure_openai_endpoint:
                errors.append(
                    "AZURE_OPENAI_ENDPOINT must be set when using Azure provider"
                )

        if not self.chroma_db_path.exists():
            errors.append(f"ChromaDB path does not exist: {self.chroma_db_path}")

        if not self.upload_dir.exists():
            errors.append(f"Upload directory does not exist: {self.upload_dir}")

        return errors

    def get_config_summary(self) -> dict:
        """Get a summary of current configuration"""
        return {
            "llm_provider": self.llm_provider,
            "google_cloud_project": self.google_cloud_project,
            "vertex_ai_location": self.vertex_ai_location,
            "has_google_api_key": bool(self.google_api_key),
            "has_azure_openai_api_key": bool(self.azure_openai_api_key),
            "azure_openai_endpoint": self.azure_openai_endpoint,
            "azure_openai_deployment_name": self.azure_openai_deployment_name,
            "elasticsearch_url": self.elasticsearch_url,
            "has_elasticsearch_api_key": bool(self.elasticsearch_api_key),
            "elasticsearch_index": self.elasticsearch_index,
            "chroma_db_path": str(self.chroma_db_path),
            "upload_dir": str(self.upload_dir),
            "embedding_model": self.embedding_model,
            "default_llm_model": self.default_llm_model,
            "max_chunk_size": self.max_chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "default_top_k": self.default_top_k,
            "web_search_timeout": self.web_search_timeout,
            "supported_extensions": list(self.supported_extensions),
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
