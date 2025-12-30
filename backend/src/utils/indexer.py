import os
import asyncio
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

from ..services.qdrant_service import QdrantService
from ..services.postgres_service import PostgresService
from ..services.embedding_service import EmbeddingService
from ..utils.text_chunker import TextChunker
from ..models.document import Document
from ..models.document_chunk import DocumentChunk


class DocumentIndexer:
    """
    Utility class for indexing documents into the RAG system
    """

    def __init__(self):
        self.qdrant_service = QdrantService()
        self.postgres_service = PostgresService()
        self.embedding_service = EmbeddingService()
        self.text_chunker = TextChunker()
        # Set the docs path relative to the project root (two levels up from src/utils)
        self.docs_path = Path(__file__).parent.parent.parent / "docs"  # Path to project root / docs

    async def initialize_services(self):
        """Initialize all required services"""
        await self.postgres_service.connect()

    async def index_single_document(self, doc_path: str, title: str = None) -> str:
        """
        Index a single document into the RAG system

        Args:
            doc_path: Path to the document file
            title: Title of the document (optional, will use filename if not provided)

        Returns:
            Document ID of the indexed document
        """
        # Read the document content
        with open(doc_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Use filename as title if not provided
        if not title:
            title = Path(doc_path).stem

        # Create document object
        doc_id = str(hash(f"{doc_path}_{datetime.now().isoformat()}"))  # Simple ID generation
        document = Document(
            id=doc_id,
            title=title,
            content=content,
            source_path=doc_path,
            chunk_count=0  # Will be updated after chunking
        )

        # Store document metadata in Postgres
        await self.postgres_service.store_document(document.dict())

        # Chunk the document content
        doc_metadata = {
            "source_path": doc_path,
            "document_id": doc_id,
            "title": title
        }
        chunks = self.text_chunker.chunk_document(content, doc_metadata)

        # Update document with chunk count
        document.chunk_count = len(chunks)
        await self.postgres_service.store_document(document.dict())

        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Create embedding for the chunk
            embedding = await self.embedding_service.create_embedding(chunk['content'])

            # Store the embedding in Qdrant
            vector_id = self.qdrant_service.store_embedding(
                text=chunk['content'],
                metadata=chunk['metadata']
            )

            # Store chunk metadata in Postgres
            chunk_obj = DocumentChunk(
                id=str(hash(f"{doc_id}_chunk_{i}")),  # Simple ID generation
                document_id=doc_id,
                content=chunk['content'],
                chunk_index=i,
                vector_id=vector_id,
                metadata=chunk['metadata']
            )

            await self.postgres_service.store_document_chunk(chunk_obj.dict())

        return doc_id

    async def index_all_documents(self, docs_directory: str = "docs") -> List[str]:
        """
        Index all markdown documents in the specified directory

        Args:
            docs_directory: Path to the directory containing documents

        Returns:
            List of document IDs that were indexed
        """
        indexed_docs = []
        docs_path = Path(docs_directory)

        # Find all markdown files in the directory and subdirectories
        markdown_files = list(docs_path.rglob("*.md")) + list(docs_path.rglob("*.mdx"))

        for md_file in markdown_files:
            try:
                print(f"Indexing {md_file}")
                doc_id = await self.index_single_document(str(md_file), title=md_file.stem)
                indexed_docs.append(doc_id)
                print(f"Successfully indexed {md_file} with ID: {doc_id}")
            except Exception as e:
                print(f"Error indexing {md_file}: {str(e)}")
                continue

        return indexed_docs

    async def index_specific_files(self, file_paths: List[str]) -> List[str]:
        """
        Index specific files provided in the list

        Args:
            file_paths: List of file paths to index

        Returns:
            List of document IDs that were indexed
        """
        indexed_docs = []
        for file_path in file_paths:
            try:
                print(f"Indexing {file_path}")
                doc_id = await self.index_single_document(file_path)
                indexed_docs.append(doc_id)
                print(f"Successfully indexed {file_path} with ID: {doc_id}")
            except Exception as e:
                print(f"Error indexing {file_path}: {str(e)}")
                continue

        return indexed_docs


async def main():
    """
    Main function to run the indexer
    """
    indexer = DocumentIndexer()
    await indexer.initialize_services()

    # Index all documents in the docs directory
    print("Starting document indexing...")
    indexed_docs = await indexer.index_all_documents()
    print(f"Completed indexing {len(indexed_docs)} documents")


if __name__ == "__main__":
    asyncio.run(main())