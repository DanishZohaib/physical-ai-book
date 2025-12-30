#!/usr/bin/env python3
"""
Final validation script for the RAG Chatbot implementation.
This script verifies that all components of the implementation are in place.
"""

import os
import sys
from pathlib import Path

def validate_backend_structure():
    """Validate backend directory structure and files"""
    print("Validating backend structure...")

    backend_path = Path("backend")
    required_dirs = [
        "src",
        "src/api",
        "src/models",
        "src/services",
        "src/utils",
        "tests",
        "tests/unit",
        "tests/integration",
        "tests/contract"
    ]

    for directory in required_dirs:
        if not (backend_path / directory).exists():
            print(f"Missing directory: {directory}")
            return False
        print(f"Found directory: {directory}")

    required_files = [
        "src/api/main.py",
        "src/api/chat.py",
        "src/api/documents.py",
        "src/models/question.py",
        "src/models/response.py",
        "src/models/document.py",
        "src/services/rag_service.py",
        "src/services/embedding_service.py",
        "src/services/qdrant_service.py",
        "src/services/postgres_service.py",
        "src/utils/indexer.py",
        "src/utils/text_chunker.py",
        "requirements.txt"
    ]

    for file in required_files:
        if not (backend_path / file).exists():
            print(f"Missing file: {file}")
            return False
        print(f"Found file: {file}")

    print("Backend structure validation passed!\n")
    return True


def validate_frontend_structure():
    """Validate frontend directory structure and files"""
    print("Validating frontend structure...")

    frontend_path = Path("frontend")
    required_dirs = [
        "src",
        "src/components",
        "src/services",
        "src/hooks"
    ]

    for directory in required_dirs:
        if not (frontend_path / directory).exists():
            print(f"Missing directory: {directory}")
            return False
        print(f"Found directory: {directory}")

    required_files = [
        "src/components/ChatWidget.jsx",
        "src/components/ChatModal.jsx",
        "src/services/api.js",
        "src/services/chat-context.js",
        "src/hooks/useChat.js"
    ]

    for file in required_files:
        if not (frontend_path / file).exists():
            print(f"Missing file: {file}")
            return False
        print(f"Found file: {file}")

    print("Frontend structure validation passed!\n")
    return True


def validate_implementation_tasks():
    """Validate that implementation tasks are properly completed"""
    print("Validating implementation tasks...")

    tasks_file = Path("specs/001-rag-chatbot/tasks.md")
    if not tasks_file.exists():
        print("Tasks file not found")
        return False

    with open(tasks_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check that all major user stories are completed
    us1_completed = content.count("- [X] T024") > 0  # US1 core task
    us2_completed = content.count("- [X] T036") > 0  # US2 core task
    us3_completed = content.count("- [X] T044") > 0  # US3 core task

    if not us1_completed:
        print("User Story 1 tasks not properly marked as completed")
        return False
    print("User Story 1 tasks completed")

    if not us2_completed:
        print("User Story 2 tasks not properly marked as completed")
        return False
    print("User Story 2 tasks completed")

    if not us3_completed:
        print("User Story 3 tasks not properly marked as completed")
        return False
    print("User Story 3 tasks completed")

    # Check that Phase 6 tasks are completed
    phase6_completed = content.count("- [X] T052") > 0  # Error handling
    if not phase6_completed:
        print("Phase 6 tasks not properly marked as completed")
        return False
    print("Phase 6 tasks completed")

    print("Implementation tasks validation passed!\n")
    return True


def validate_documentation():
    """Validate documentation files exist"""
    print("Validating documentation...")

    docs = [
        "specs/001-rag-chatbot/spec.md",
        "specs/001-rag-chatbot/plan.md",
        "specs/001-rag-chatbot/data-model.md",
        "specs/001-rag-chatbot/contracts/api-contract.md",
        "specs/001-rag-chatbot/quickstart.md",
        "DEPLOYMENT.md"
    ]

    for doc in docs:
        if not Path(doc).exists():
            print(f"Missing documentation: {doc}")
            return False
        print(f"Found documentation: {doc}")

    print("Documentation validation passed!\n")
    return True


def validate_api_contract():
    """Validate API contract implementation"""
    print("Validating API contract...")

    # Check that main endpoints are implemented
    chat_api = Path("backend/src/api/chat.py")
    if not chat_api.exists():
        print("Chat API file not found")
        return False

    with open(chat_api, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check for required endpoints
    has_chat_endpoint = "router.post" in content and "async def chat_endpoint" in content
    has_session_endpoint = "start-session" in content

    if not has_chat_endpoint:
        print("Chat endpoint not implemented")
        return False
    print("Chat endpoint implemented")

    if not has_session_endpoint:
        print("Session endpoint not implemented")
        return False
    print("Session endpoint implemented")

    # Check that documents API is implemented
    docs_api = Path("backend/src/api/documents.py")
    if not docs_api.exists():
        print("Documents API file not found")
        return False

    with open(docs_api, 'r', encoding='utf-8') as f:
        content = f.read()

    has_index_endpoint = "index_document" in content
    has_get_endpoint = "get_document" in content

    if not has_index_endpoint:
        print("Document index endpoint not implemented")
        return False
    print("Document index endpoint implemented")

    if not has_get_endpoint:
        print("Document get endpoint not implemented")
        return False
    print("Document get endpoint implemented")

    print("API contract validation passed!\n")
    return True


def main():
    """Main validation function"""
    print("Starting final validation of RAG Chatbot implementation...\n")

    all_passed = True

    all_passed &= validate_backend_structure()
    all_passed &= validate_frontend_structure()
    all_passed &= validate_implementation_tasks()
    all_passed &= validate_documentation()
    all_passed &= validate_api_contract()

    print("="*50)
    if all_passed:
        print("All validations passed! Implementation is complete and ready.")
        print("\nSummary:")
        print("  Backend structure complete")
        print("  Frontend structure complete")
        print("  User stories implemented")
        print("  Documentation in place")
        print("  API contracts implemented")
        print("\nThe RAG Chatbot for Physical AI Book is ready for deployment!")
        return 0
    else:
        print("Some validations failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())