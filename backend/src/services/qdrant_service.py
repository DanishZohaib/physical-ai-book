from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any
import uuid
from dotenv import load_dotenv
import os

load_dotenv()

class QdrantService:
    def __init__(self):
        # Check if we're running in local mode (for testing without remote Qdrant)
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

        # Try to connect to remote Qdrant, fallback to in-memory for testing
        try:
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=os.getenv("QDRANT_API_KEY")
            )
            # Test the connection
            self.client.get_collections()
            self.remote_mode = True
        except:
            # Fallback to in-memory client for local testing
            self.client = QdrantClient(":memory:")
            self.remote_mode = False

        self.collection_name = "book_content"
        self._init_collection()

    def _init_collection(self):
        """Initialize the Qdrant collection for book content"""
        try:
            # Check if collection exists
            self.client.get_collection(self.collection_name)
        except:
            # Create collection if it doesn't exist
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)  # Assuming OpenAI embeddings
            )

    def store_embedding(self, text: str, metadata: Dict[str, Any]) -> str:
        """Store a text embedding in Qdrant"""
        vector_id = str(uuid.uuid4())

        # In a real implementation, we would generate the embedding here
        # For now, we'll use a placeholder
        embedding = [0.0] * 1536  # Placeholder - would be actual embedding

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=vector_id,
                    vector=embedding,
                    payload={
                        "text": text,
                        **metadata
                    }
                )
            ]
        )

        return vector_id

    def search_similar(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar content based on embedding"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )

        return [
            {
                "id": result.id,
                "text": result.payload.get("text", ""),
                "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                "score": result.score
            }
            for result in results
        ]

    def delete_by_document_id(self, document_id: str):
        """Delete all embeddings associated with a document"""
        # This would require filtering by document_id in payload
        # Implementation would depend on how document_id is stored in metadata
        pass