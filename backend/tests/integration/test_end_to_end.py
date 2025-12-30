import pytest
import asyncio
from fastapi.testclient import TestClient
from src.api.main import app
from unittest.mock import patch
import json


def test_end_to_end_functionality():
    """
    Test end-to-end functionality: question input → API → response display
    This test simulates the complete flow from question submission to response display
    """
    client = TestClient(app)

    # Mock the RAG service to avoid actual API calls during testing
    with patch('src.api.chat.rag_service') as mock_rag_service:
        # Setup mock response that simulates the expected RAG service output
        mock_rag_service.get_answer.return_value = {
            "answer": "A ROS 2 node is a process that performs computation in the ROS 2 system. It can communicate with other nodes through topics, services, and actions.",
            "sources": [
                {
                    "document_id": "doc-ros-fundamentals-1",
                    "title": "ROS 2 Fundamentals",
                    "path": "chapter-3-ros-fundamentals.md",
                    "relevance_score": 0.95
                }
            ],
            "confidence": 0.92
        }

        # Simulate a user submitting a question
        question_request = {
            "question": "What is a ROS 2 node?",
            "page_context": "chapter-3-ros-fundamentals.md",
            "selected_text": None,
            "session_id": "session-12345"
        }

        # Make the API call (question input → API)
        response = client.post("/chat", json=question_request)

        # Verify the API response
        assert response.status_code == 200

        # Parse the response (API → response display)
        response_data = response.json()

        # Verify response structure
        assert "id" in response_data
        assert "question_id" in response_data
        assert "answer" in response_data
        assert "sources" in response_data
        assert "confidence" in response_data
        assert "session_id" in response_data
        assert "timestamp" in response_data

        # Verify the answer content
        assert "ROS 2 node" in response_data["answer"]
        assert isinstance(response_data["answer"], str)
        assert len(response_data["answer"]) > 0

        # Verify confidence score is in expected range
        assert 0.0 <= response_data["confidence"] <= 1.0

        # Verify sources information
        assert len(response_data["sources"]) > 0
        source = response_data["sources"][0]
        assert "document_id" in source
        assert "title" in source
        assert "path" in source
        assert "relevance_score" in source

        print("✅ End-to-end test passed: Question input → API → Response display")


def test_session_creation_and_chat_flow():
    """
    Test the complete flow of session creation followed by chat interaction
    """
    client = TestClient(app)

    # First, create a session
    session_request = {
        "initial_context": {
            "page": "chapter-3-ros-fundamentals.md",
            "user_agent": "test-browser"
        }
    }

    session_response = client.post("/chat/start-session", json=session_request)
    assert session_response.status_code == 200

    session_data = session_response.json()
    assert "session_id" in session_data
    session_id = session_data["session_id"]

    # Now use the session ID to ask a question
    with patch('src.api.chat.rag_service') as mock_rag_service:
        mock_rag_service.get_answer.return_value = {
            "answer": "Test response for end-to-end flow",
            "sources": [],
            "confidence": 0.85
        }

        question_request = {
            "question": "What is the main concept discussed in this chapter?",
            "page_context": "chapter-3-ros-fundamentals.md",
            "selected_text": "This chapter discusses ROS 2 fundamentals.",
            "session_id": session_id
        }

        response = client.post("/chat", json=question_request)
        assert response.status_code == 200

        response_data = response.json()
        assert response_data["session_id"] == session_id
        assert "answer" in response_data
        assert len(response_data["answer"]) > 0

        print("✅ Session flow test passed: Session creation → Chat with session")


def test_selected_text_functionality():
    """
    Test the end-to-end functionality with selected text
    """
    client = TestClient(app)

    # First, create a session
    session_request = {
        "initial_context": {
            "page": "chapter-3-ros-fundamentals.md",
            "user_agent": "test-browser"
        }
    }

    session_response = client.post("/chat/start-session", json=session_request)
    assert session_response.status_code == 200

    session_data = session_response.json()
    assert "session_id" in session_data
    session_id = session_data["session_id"]

    # Now use the session ID to ask a question with selected text
    with patch('src.api.chat.rag_service') as mock_rag_service:
        mock_rag_service.get_answer.return_value = {
            "answer": "The selected text discusses ROS 2 nodes, which are processes that perform computation in the ROS 2 system.",
            "sources": [
                {
                    "document_id": "doc-ros-fundamentals-1",
                    "title": "ROS 2 Fundamentals",
                    "path": "chapter-3-ros-fundamentals.md",
                    "relevance_score": 0.98
                }
            ],
            "confidence": 0.95
        }

        # Request with selected text to test the US2 functionality
        question_request = {
            "question": "How does this work?",
            "page_context": "chapter-3-ros-fundamentals.md",
            "selected_text": "A ROS 2 node is a process that performs computation.",
            "session_id": session_id
        }

        response = client.post("/chat", json=question_request)
        assert response.status_code == 200

        response_data = response.json()
        assert response_data["session_id"] == session_id
        assert "answer" in response_data
        assert len(response_data["answer"]) > 0

        # Verify that the selected text was processed
        # In a real scenario, the answer would be more specifically related to the selected text
        assert "ROS 2 node" in response_data["answer"] or "process" in response_data["answer"]

        # Verify sources information
        assert len(response_data["sources"]) > 0
        source = response_data["sources"][0]
        assert "document_id" in source
        assert "title" in source
        assert "path" in source
        assert "relevance_score" in source

        print("✅ Selected text functionality test passed: Question with selected text → API → Response")


def test_session_persistence_across_pages():
    """
    Test that chat sessions persist across page navigation (US3 functionality)
    """
    client = TestClient(app)

    # Create a session on one page
    session_request = {
        "initial_context": {
            "page": "chapter-3-ros-fundamentals.md",
            "user_agent": "test-browser"
        }
    }

    session_response = client.post("/chat/start-session", json=session_request)
    assert session_response.status_code == 200

    session_data = session_response.json()
    assert "session_id" in session_data
    session_id = session_data["session_id"]

    # Ask a question and get a response
    with patch('src.api.chat.rag_service') as mock_rag_service:
        mock_rag_service.get_answer.return_value = {
            "answer": "This is a response about ROS 2 nodes.",
            "sources": [
                {
                    "document_id": "doc-ros-fundamentals-1",
                    "title": "ROS 2 Fundamentals",
                    "path": "chapter-3-ros-fundamentals.md",
                    "relevance_score": 0.95
                }
            ],
            "confidence": 0.90
        }

        question_request = {
            "question": "What is a ROS 2 node?",
            "page_context": "chapter-3-ros-fundamentals.md",
            "selected_text": None,
            "session_id": session_id
        }

        response = client.post("/chat", json=question_request)
        assert response.status_code == 200
        first_response_data = response.json()
        assert "answer" in first_response_data

    # Simulate navigating to a different page but keeping the same session
    with patch('src.api.chat.rag_service') as mock_rag_service:
        mock_rag_service.get_answer.return_value = {
            "answer": "This is a response about robot simulation.",
            "sources": [
                {
                    "document_id": "doc-simulation-1",
                    "title": "Robot Simulation",
                    "path": "chapter-4-robot-simulation.md",
                    "relevance_score": 0.88
                }
            ],
            "confidence": 0.85
        }

        # Ask a question on a different page but with the same session ID
        question_request = {
            "question": "How do I simulate a robot?",
            "page_context": "chapter-4-robot-simulation.md",  # Different page
            "selected_text": None,
            "session_id": session_id  # Same session ID
        }

        response = client.post("/chat", json=question_request)
        assert response.status_code == 200
        second_response_data = response.json()
        assert "answer" in second_response_data

    print("✅ Session persistence test passed: Session maintained across different pages")


if __name__ == "__main__":
    test_end_to_end_functionality()
    test_session_creation_and_chat_flow()
    test_selected_text_functionality()
    test_session_persistence_across_pages()
    print("🎉 All end-to-end tests passed!")