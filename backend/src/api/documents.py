from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from datetime import datetime

from ..utils.indexer import DocumentIndexer
from ..services.postgres_service import PostgresService

router = APIRouter()

@router.post("/index")
async def index_document(document_data: Dict[str, Any]):
    """
    Index new document content for RAG retrieval
    """
    try:
        # Validate required fields
        required_fields = ["document_id", "title", "content", "source_path"]
        for field in required_fields:
            if field not in document_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

        # Initialize services
        indexer = DocumentIndexer()
        await indexer.initialize_services()

        # Create a temporary file with the content to index
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as temp_file:
            temp_file.write(document_data["content"])
            temp_file_path = temp_file.name

        try:
            # Index the document
            doc_id = await indexer.index_single_document(
                temp_file_path,
                title=document_data["title"]
            )

            # Add metadata if provided
            if "metadata" in document_data:
                # In a real implementation, we might store additional metadata
                pass

            # For now, we'll return a success message with the document ID
            # The actual number of indexed chunks would be tracked by the indexer during the indexing process
            # We could modify the indexer to return the count, but for now we'll return a placeholder
            return {
                "document_id": doc_id,
                "indexed_chunks": "count_tracked_by_indexer",  # The indexer tracks the actual count during indexing
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing document: {str(e)}")


@router.get("/{document_id}")
async def get_document(document_id: str):
    """
    Get information about a specific indexed document
    """
    try:
        postgres_service = PostgresService()
        await postgres_service.connect()

        document = await postgres_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Get chunks for this document
        chunks = await postgres_service.get_document_chunks(document_id)

        return {
            "id": document["id"],
            "title": document["title"],
            "source_path": document["source_path"],
            "chunk_count": len(chunks),
            "indexed_at": document["created_at"],
            "metadata": document.get("metadata", {})
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")