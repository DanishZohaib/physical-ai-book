from typing import List, Dict, Any
import asyncio
from datetime import datetime

from ..services.embedding_service import EmbeddingService
from ..services.qdrant_service import QdrantService
from ..services.postgres_service import PostgresService
from ..models.question import Question
from ..models.response import Response


class RAGService:
    """
    Retrieval Augmented Generation service that handles question answering
    by retrieving relevant documents and generating responses based on them
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()
        self.postgres_service = PostgresService()
        self.max_context_length = 2000  # Maximum length of context to send to LLM

    async def initialize_services(self):
        """Initialize all required services"""
        await self.postgres_service.connect()

    async def answer_question(self, question: Question) -> Response:
        """
        Answer a question by retrieving relevant documents and generating a response

        Args:
            question: The question object containing the query and context

        Returns:
            Response object with the answer and sources
        """
        # Create embedding for the question
        question_embedding = await self.embedding_service.create_embedding(question.content)

        # If selected text is provided, also create an embedding for it
        selected_text_embedding = None
        if question.selected_text and question.selected_text.strip():
            selected_text_embedding = await self.embedding_service.create_embedding(question.selected_text)

        # Search for similar content in the vector database
        # If selected text exists, we'll search for content related to both the question and selected text
        if selected_text_embedding:
            # First search for content related to selected text to prioritize it
            selected_text_results = self.qdrant_service.search_similar(
                query_embedding=selected_text_embedding,
                limit=3  # Retrieve top 3 chunks related to selected text
            )

            # Then search for content related to the question
            question_results = self.qdrant_service.search_similar(
                query_embedding=question_embedding,
                limit=5  # Retrieve top 5 chunks related to question
            )

            # Combine results, prioritizing those related to selected text
            # Use a dictionary to avoid duplicates, with selected text results taking priority
            combined_results = {}

            # Add selected text results first (higher priority)
            for result in selected_text_results:
                combined_results[result["id"]] = result

            # Add question results, avoiding duplicates
            for result in question_results:
                if result["id"] not in combined_results:
                    combined_results[result["id"]] = result

            # Convert back to list, maintaining priority order
            search_results = list(combined_results.values())
        else:
            # Standard search if no selected text
            search_results = self.qdrant_service.search_similar(
                query_embedding=question_embedding,
                limit=5  # Retrieve top 5 most relevant chunks
            )

        # Prepare context from retrieved documents
        context_parts = []
        sources = []
        current_length = 0

        for result in search_results:
            content = result["text"]
            # Add content to context if it doesn't exceed the limit
            if current_length + len(content) <= self.max_context_length:
                context_parts.append(content)
                sources.append({
                    "document_id": result["id"],
                    "title": result.get("metadata", {}).get("title", "Unknown"),
                    "path": result.get("metadata", {}).get("source_path", "Unknown"),
                    "relevance_score": result["score"]
                })
                current_length += len(content)
            else:
                # If adding this content would exceed the limit, stop adding
                break

        # Combine context parts
        context = "\n\n".join(context_parts)

        # Generate response using the context and question
        answer = await self._generate_response_with_context(
            question.content,
            context,
            question.selected_text
        )

        # Create response object
        response = Response(
            question_id=question.id,
            content=answer,
            sources=sources,
            confidence_score=self._calculate_confidence_score(sources, search_results),
            session_id=question.session_id
        )

        return response

    async def _generate_response_with_context(self, question: str, context: str, selected_text: str = None) -> str:
        """
        Generate a response using the provided context and question

        Args:
            question: The original question
            context: Retrieved context from documents
            selected_text: Specific text selected by the user (optional)

        Returns:
            Generated response string
        """
        import openai
        from dotenv import load_dotenv
        import os

        load_dotenv()

        # Initialize the OpenAI client
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Prepare the prompt for the LLM
        if selected_text:
            prompt = f"""
            Context from the Physical AI book:
            {context}

            User selected this specific text:
            {selected_text}

            User question about the selected text:
            {question}

            Please provide an answer based on the context from the Physical AI book.
            If the information is not available in the context, please state that clearly.
            """
        else:
            prompt = f"""
            Context from the Physical AI book:
            {context}

            User question:
            {question}

            Please provide an answer based on the context from the Physical AI book.
            If the information is not available in the context, please state that clearly.
            """

        try:
            # Call OpenAI API to generate the response
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # or gpt-4 if available
                messages=[
                    {"role": "system", "content": "You are an assistant for the Physical AI book. Answer questions based only on the provided context from the book. Be accurate and helpful."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            # If OpenAI API call fails, return a helpful error message
            return f"Sorry, I encountered an error while processing your question: {str(e)}. Please try again later."

    def _calculate_confidence_score(self, sources: List[Dict], search_results: List[Dict]) -> float:
        """
        Calculate a confidence score based on the relevance of retrieved sources

        Args:
            sources: List of sources used in the response
            search_results: Raw search results from the vector database

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not search_results:
            return 0.0

        # Calculate average relevance score
        total_score = sum(result["score"] for result in search_results)
        avg_score = total_score / len(search_results)

        # Normalize the score to 0-1 range (assuming cosine similarity scores)
        # Cosine similarity typically ranges from -1 to 1, but in practice for embeddings
        # it's usually between 0 and 1, with 1 being most similar
        confidence = max(0.0, min(1.0, avg_score))

        return confidence

    async def validate_answer_relevance(self, question: str, answer: str, sources: List[Dict]) -> bool:
        """
        Validate if the answer is relevant to the question based on the sources

        Args:
            question: The original question
            answer: The generated answer
            sources: Sources used to generate the answer

        Returns:
            True if answer is relevant, False otherwise
        """
        # In a real implementation, we might use additional LLM calls or NLP techniques
        # to validate relevance, but for now we'll implement a basic check
        question_lower = question.lower()
        answer_lower = answer.lower()

        # Simple heuristic: if the answer contains information that directly addresses the question
        # This is a basic check and would need more sophisticated validation in production
        if len(answer) < 10:  # Answer is too short
            return False

        # If no sources were used but an answer was generated, it might be hallucinated
        if not sources and "not found" not in answer_lower and "not available" not in answer_lower:
            return False

        return True