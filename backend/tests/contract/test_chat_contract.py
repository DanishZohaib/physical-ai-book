import pytest
import asyncio
from fastapi.testclient import TestClient
from src.api.main import app
from src.services.rag_service import RAGService
from unittest.mock import AsyncMock, patch


@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


@pytest.mark.asyncio
async def test_chat_endpoint_contract():
    """Test the contract for the POST /chat endpoint"""
    client = TestClient(app)

    # Mock the RAG service to avoid actual API calls during testing
    with patch('src.api.chat.rag_service') as mock_rag_service:
        # Setup mock response
        mock_rag_service.get_answer.return_value = {
            "answer": "This is a test response based on book content.",
            "sources": [
                {
                    "document_id": "test-doc-1",
                    "title": "Test Document",
                    "path": "test-document.md",
                    "relevance_score": 0.85
                }
            ],
            "confidence": 0.92
        }

        # Test request data matching the contract
        request_data = {
            "question": "What is a ROS 2 node?",
            "page_context": "chapter-3-ros-fundamentals.md",
            "selected_text": "A ROS 2 node is a process that performs computation.",
            "session_id": "test-session-id"
        }

        # Make the request
        response = client.post("/chat", json=request_data)

        # Verify response structure and status
        assert response.status_code == 200

        # Parse response
        data = response.json()

        # Verify response structure matches contract
        assert "id" in data
        assert "question_id" in data
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data
        assert "session_id" in data
        assert "timestamp" in data

        # Verify data types and content
        assert isinstance(data["id"], str)
        assert isinstance(data["question_id"], str)
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["confidence"], (int, float))
        assert isinstance(data["session_id"], str)
        assert isinstance(data["timestamp"], str)  # ISO format timestamp

        # Verify sources structure
        if data["sources"]:
            source = data["sources"][0]
            assert "document_id" in source
            assert "title" in source
            assert "path" in source
            assert "relevance_score" in source


@pytest.mark.asyncio
async def test_chat_endpoint_validation():
    """Test validation of the POST /chat endpoint"""
    client = TestClient(app)

    # Test with invalid request format
    invalid_request = {
        "invalid_field": "invalid_value"
    }

    response = client.post("/chat", json=invalid_request)

    # Should return 400 for invalid request
    assert response.status_code in [400, 422]  # 422 is also valid for validation errors


@pytest.mark.asyncio
async def test_chat_endpoint_missing_fields():
    """Test the endpoint with missing required fields"""
    client = TestClient(app)

    # Test with minimal required fields
    minimal_request = {
        "question": "What is this?",
        "session_id": "test-session-id"
    }

    with patch('src.api.chat.rag_service') as mock_rag_service:
        mock_rag_service.get_answer.return_value = {
            "answer": "Test response",
            "sources": [],
            "confidence": 0.8
        }

        response = client.post("/chat", json=minimal_request)

        # Should succeed with minimal valid request
        assert response.status_code == 200


def test_chat_endpoint_method():
    """Test that the endpoint only accepts POST requests"""
    client = TestClient(app)

    # Test GET method (should not be allowed)
    response = client.get("/chat")
    assert response.status_code in [405, 404]  # Method not allowed or not found
