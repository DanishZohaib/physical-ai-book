import openai
from typing import List
from dotenv import load_dotenv
import os
import hashlib

load_dotenv()

class EmbeddingService:
    def __init__(self):
        # Check if we have a real API key, otherwise use mock mode
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.mock_mode = not self.api_key or self.api_key.startswith("sk-LOCAL")

    async def create_embedding(self, text: str) -> List[float]:
        """Create an embedding for the given text using OpenAI API or mock if no key"""
        if self.mock_mode:
            # Create a deterministic mock embedding based on the text content
            # This ensures consistent results for testing
            text_hash = hashlib.sha256(text.encode()).hexdigest()

            # Generate a 1536-dimensional vector based on the hash
            embedding = []
            for i in range(0, 1536 * 2, 2):
                if i + 1 < len(text_hash):
                    # Take pairs of hex characters and convert to float
                    hex_pair = text_hash[i:i+2]
                    val = int(hex_pair, 16) / 255.0  # Normalize to 0-1 range
                    embedding.append(val)
                else:
                    # Pad with zeros if needed
                    embedding.append(0.0)

            # Trim or pad to exactly 1536 dimensions
            if len(embedding) < 1536:
                embedding.extend([0.0] * (1536 - len(embedding)))
            elif len(embedding) > 1536:
                embedding = embedding[:1536]

            return embedding
        else:
            # Use real OpenAI API with newer format and timeout handling
            try:
                client = openai.OpenAI(api_key=self.api_key)
                response = client.embeddings.create(
                    input=text,
                    model="text-embedding-ada-002",  # Using OpenAI's standard embedding model
                    timeout=30  # 30 second timeout
                )
                embedding = response.data[0].embedding
                return embedding
            except Exception as e:
                print(f"Error creating embedding: {e}")
                # Return a default embedding in case of error
                return [0.0] * 1536

    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a batch of texts"""
        embeddings = []
        for text in texts:
            embedding = await self.create_embedding(text)
            embeddings.append(embedding)
        return embeddings

    async def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)