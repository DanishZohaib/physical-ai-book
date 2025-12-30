from fastapi import FastAPI
from . import chat, documents
from datetime import datetime
import os

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify service availability"""
    import asyncio
    from ..services.qdrant_service import QdrantService
    from ..services.postgres_service import PostgresService
    from ..services.embedding_service import EmbeddingService

    dependencies_status = {}

    # Check Qdrant connection
    try:
        qdrant_service = QdrantService()
        # If initialization succeeded, connection is good
        dependencies_status["qdrant"] = "connected"
    except Exception as e:
        dependencies_status["qdrant"] = f"error: {str(e)}"

    # Check Postgres connection
    try:
        postgres_service = PostgresService()
        await postgres_service.connect()
        dependencies_status["postgres"] = "connected"
    except Exception as e:
        dependencies_status["postgres"] = f"error: {str(e)}"

    # Check Embedding service (OpenAI)
    try:
        embedding_service = EmbeddingService()
        # Test with a simple text
        test_embedding = await embedding_service.create_embedding("test")
        if len(test_embedding) > 0:
            dependencies_status["openai"] = "connected"
        else:
            dependencies_status["openai"] = "error: invalid response"
    except Exception as e:
        dependencies_status["openai"] = f"error: {str(e)}"

    # Determine overall status
    all_connected = all(status == "connected" for status in dependencies_status.values())
    overall_status = "healthy" if all_connected else "degraded"

    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "dependencies": dependencies_status
    }

# Include other API routes
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])