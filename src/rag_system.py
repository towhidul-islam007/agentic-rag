"""Main RAG system implementation"""

import asyncio
import logging

from pathlib import Path
from typing import Any, Dict, List

from haystack import Document, Pipeline
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from langchain_core.messages import HumanMessage

from src.agentic_rag import create_graph
from src.utils import (
    get_document_count_sync,
    get_document_embedder,
    get_document_store,
    save_document_store,
)

logger = logging.getLogger(__name__)


class AgenticRAG:
    """Main Agentic RAG system"""

    def __init__(self, gemini_model: str = "gemini-2.5-flash") -> None:
        self.gemini_model = gemini_model

        # Initialize document processing pipeline
        self._build_indexing_pipeline()

        # Create the workflow graph
        self.app = create_graph(gemini_model)

    def _build_indexing_pipeline(self) -> None:
        """Build document indexing pipeline"""
        document_store = get_document_store()

        # Initialize document embedder (Google embeddings preferred)
        doc_embedder = get_document_embedder()

        self.indexing_pipeline = Pipeline()

        # Add components
        self.indexing_pipeline.add_component(
            "splitter",
            DocumentSplitter(split_by="sentence", split_length=3, split_overlap=1),
        )
        self.indexing_pipeline.add_component(
            "metadata_cleaner", self._create_metadata_cleaner()
        )
        self.indexing_pipeline.add_component("embedder", doc_embedder)
        self.indexing_pipeline.add_component(
            "writer", DocumentWriter(document_store=document_store)
        )

        # Connect components
        self.indexing_pipeline.connect("splitter", "metadata_cleaner")
        self.indexing_pipeline.connect("metadata_cleaner", "embedder")
        self.indexing_pipeline.connect("embedder", "writer")

    def _create_metadata_cleaner(self) -> object:
        """Create a component to clean metadata for ChromaDB compatibility"""
        from typing import Any, Dict, List

        from haystack import component

        @component
        class MetadataCleaner:
            @component.output_types(documents=List[Document])
            def run(self, documents: List[Document]) -> Dict[str, Any]:
                """Clean metadata to only include supported types"""
                cleaned_docs = []
                for doc in documents:
                    # Create new metadata with only supported types
                    cleaned_meta = {}
                    for key, value in doc.meta.items():
                        if isinstance(value, (str, int, float, bool)):
                            cleaned_meta[key] = value
                        elif key == "_split_overlap":
                            # Convert to supported type
                            cleaned_meta[key] = (
                                int(value) if isinstance(value, (int, float)) else 0
                            )
                        # Skip unsupported types

                    # Create new document with cleaned metadata
                    cleaned_doc = Document(
                        content=doc.content,
                        meta=cleaned_meta,
                        embedding=doc.embedding if hasattr(doc, "embedding") else None,
                    )
                    cleaned_docs.append(cleaned_doc)

                return {"documents": cleaned_docs}

        return MetadataCleaner()

    async def add_documents(self, file_paths: List[str]) -> bool:
        """Add documents to the vector store"""
        try:
            documents = []

            for file_path in file_paths:
                path = Path(file_path)

                if path.suffix.lower() == ".pdf":
                    converter = PyPDFToDocument()
                    result = converter.run(sources=[file_path])
                    documents.extend(result["documents"])

                elif path.suffix.lower() in [".txt", ".md"]:
                    converter = TextFileToDocument()
                    result = converter.run(sources=[file_path])
                    documents.extend(result["documents"])

                else:
                    logger.warning(f"Unsupported file type: {path.suffix}")

            if documents:
                # Run the pipeline in executor to keep the async interface
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self.indexing_pipeline.run,
                    {"splitter": {"documents": documents}},
                )
                await save_document_store()
                logger.info(f"Successfully indexed {len(documents)} documents")
                return True

        except Exception as e:
            logger.error(f"Error adding documents: {e}")

        return False

    async def query(self, question: str) -> Dict[str, Any]:
        """Query the agentic RAG system"""
        try:
            # Prepare initial state
            inputs = {
                "question": question,
                "messages": [HumanMessage(content=question)],
                "query_analysis": {},
                "route_decision": "vectorstore",
                "documents": [],
                "document_relevance_scores": [],
                "filtered_documents": [],
                "no_documents_found": False,
                "web_results": [],
                "web_search_query": "",
                "context": "",
                "generation": "",
                "relevance_grade": "relevant",
                "hallucination_grade": "no",
                "answer_grade": "useful",
                "loop_count": 0,
                "max_loops": 3,
            }

            # Run the workflow with recursion limit asynchronously
            result = await self.app.ainvoke(inputs, config={"recursion_limit": 10})

            return {
                "response": result.get("generation", "No response generated"),
                "context": result.get("context", ""),
                "route_decision": result.get("route_decision", "unknown"),
                "num_documents": len(result.get("filtered_documents", [])),
                "num_web_results": len(result.get("web_results", [])),
                "query_analysis": result.get("query_analysis", {}),
                "relevance_grade": result.get("relevance_grade", ""),
                "hallucination_grade": result.get("hallucination_grade", ""),
                "answer_grade": result.get("answer_grade", ""),
            }

        except Exception as e:
            logger.error(f"Error in query: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your query.",
                "context": "",
                "route_decision": "error",
                "num_documents": 0,
                "num_web_results": 0,
                "query_analysis": {},
                "relevance_grade": "",
                "hallucination_grade": "",
                "answer_grade": "",
            }

    def get_document_count(self) -> int:
        """Get the number of documents in the store"""
        return get_document_count_sync()
