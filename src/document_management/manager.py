"""Document Manager for handling file uploads and processing"""

import asyncio
import logging

from pathlib import Path
from typing import List, Optional

import streamlit as st

from docx import Document as DocxDocument

from config import get_settings

# Get settings instance
settings = get_settings()

logger = logging.getLogger(__name__)


class DocumentManager:
    """Handles document upload and processing for the RAG system"""

    def __init__(self, upload_dir: Optional[str] = None) -> None:
        """Initialize the document manager.

        Args:
            upload_dir: The upload directory path. If None, uses default from settings.
        """
        if upload_dir is None:
            upload_dir = str(settings.upload_dir)
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)

        # Supported file types
        self.supported_extensions = settings.supported_extensions

    async def save_uploaded_files(self, uploaded_files: List) -> List[str]:
        """Save uploaded files and return their paths.

        Args:
            uploaded_files: The uploaded_files parameter.

        Returns:
            List[str]: String result.
        """
        saved_paths = []

        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                # Check file extension
                file_extension = Path(uploaded_file.name).suffix.lower()

                if file_extension not in self.supported_extensions:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
                    continue

                # Save file
                file_path = self.upload_dir / uploaded_file.name

                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, self._write_file, file_path, uploaded_file.getbuffer()
                    )

                    saved_paths.append(str(file_path))
                    logger.info(f"Saved file: {file_path}")

                except Exception as e:
                    logger.error(f"Error saving file {uploaded_file.name}: {e}")
                    st.error(f"Error saving file {uploaded_file.name}: {e}")

        return saved_paths

    def _write_file(self, file_path: Path, buffer: bytes) -> None:
        """Helper method to write file synchronously.

        Args:
            file_path: Path to the file or directory.
            buffer: The buffer parameter.
        """
        with Path.open(file_path, "wb") as f:
            f.write(buffer)

    async def process_docx_files(self, file_paths: List[str]) -> List[str]:
        """Convert DOCX files to text files for processing.

        Args:
            file_paths: Path to the file or directory.

        Returns:
            List[str]: String result.
        """
        processed_paths = []

        for file_path in file_paths:
            path = Path(file_path)

            if path.suffix.lower() == ".docx":
                try:
                    # Read DOCX file asynchronously
                    loop = asyncio.get_event_loop()
                    text_content = await loop.run_in_executor(
                        None, self._process_docx_file, file_path
                    )

                    # Save as text file
                    text_file_path = path.with_suffix(".txt")
                    await loop.run_in_executor(
                        None, self._write_text_file, text_file_path, text_content
                    )

                    processed_paths.append(str(text_file_path))
                    logger.info(f"Converted DOCX to text: {text_file_path}")

                except Exception as e:
                    logger.error(f"Error processing DOCX file {file_path}: {e}")
                    st.error(f"Error processing DOCX file {path.name}: {e}")
            else:
                processed_paths.append(file_path)

        return processed_paths

    def _process_docx_file(self, file_path: str) -> str:
        """Helper method to process DOCX file synchronously.

        Args:
            file_path: Path to the file or directory.

        Returns:
            str: String result.
        """
        doc = DocxDocument(file_path)
        text_content = [
            paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()
        ]
        return "\n\n".join(text_content)

    def _write_text_file(self, file_path: Path, content: str) -> None:
        """Helper method to write text file synchronously.

        Args:
            file_path: Path to the file or directory.
            content: The content parameter.
        """
        with Path.open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    async def get_uploaded_files(self) -> List[str]:
        """Get list of uploaded files.

        Returns:
            List[str]: String result.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_uploaded_files_sync)

    def _get_uploaded_files_sync(self) -> List[str]:
        """Synchronous helper for getting uploaded files.

        Returns:
            List[str]: String result.
        """
        return [
            str(file_path)
            for file_path in self.upload_dir.glob("*")
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.supported_extensions
            )
        ]

    async def delete_file(self, file_path: str) -> bool:
        """Delete a file.

        Args:
            file_path: Path to the file or directory.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, Path(file_path).unlink)
            logger.info(f"Deleted file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False

    async def clear_all_files(self) -> bool:
        """Clear all uploaded files.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._clear_all_files_sync)
            logger.info("Cleared all uploaded files")
            return True
        except Exception as e:
            logger.error(f"Error clearing files: {e}")
            return False

    def _clear_all_files_sync(self) -> None:
        """Synchronous helper for clearing all files."""
        for file_path in self.upload_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()
