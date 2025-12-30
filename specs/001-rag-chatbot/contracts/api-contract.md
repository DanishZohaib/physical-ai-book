# API Contract: RAG Chatbot Service

## Chat API

### POST /chat
Submit a question and receive an AI-generated response based on book content.

**Request**:
```json
{
  "question": "What is a ROS 2 node?",
  "page_context": "chapter-3-ros-fundamentals.md",
  "selected_text": "A ROS 2 node is a process that performs computation.",
  "session_id": "uuid-string"
}
```

**Response** (200 OK):
```json
{
  "id": "response-uuid",
  "question_id": "question-uuid",
  "answer": "A ROS 2 node is a process that performs computation in the ROS 2 system...",
  "sources": [
    {
      "document_id": "doc-uuid",
      "title": "ROS 2 Fundamentals",
      "path": "chapter-3-ros-fundamentals.md",
      "relevance_score": 0.85
    }
  ],
  "confidence": 0.92,
  "session_id": "uuid-string",
  "timestamp": "2025-01-08T10:30:00Z"
}
```

**Errors**:
- 400: Invalid request format
- 429: Rate limit exceeded
- 500: Internal server error

### POST /chat/start-session
Start a new chat session.

**Request**:
```json
{
  "initial_context": {
    "page": "chapter-3-ros-fundamentals.md",
    "user_agent": "docusaurus-client"
  }
}
```

**Response** (201 Created):
```json
{
  "session_id": "new-session-uuid",
  "created_at": "2025-01-08T10:30:00Z",
  "expires_at": "2025-01-08T11:30:00Z"
}
```

### GET /health
Check service health and availability.

**Response** (200 OK):
```json
{
  "status": "healthy",
  "timestamp": "2025-01-08T10:30:00Z",
  "dependencies": {
    "openai": "connected",
    "qdrant": "connected",
    "postgres": "connected"
  }
}
```

## Document Management API

### POST /documents/index
Index new document content for RAG retrieval.

**Request**:
```json
{
  "document_id": "doc-uuid",
  "title": "ROS 2 Fundamentals",
  "content": "Full text content of the document...",
  "source_path": "chapter-3-ros-fundamentals.md",
  "metadata": {
    "chapter": 3,
    "tags": ["ros2", "fundamentals"]
  }
}
```

**Response** (201 Created):
```json
{
  "document_id": "doc-uuid",
  "indexed_chunks": 5,
  "status": "completed",
  "timestamp": "2025-01-08T10:30:00Z"
}
```

### GET /documents/{document_id}
Get information about a specific indexed document.

**Response** (200 OK):
```json
{
  "id": "doc-uuid",
  "title": "ROS 2 Fundamentals",
  "source_path": "chapter-3-ros-fundamentals.md",
  "chunk_count": 5,
  "indexed_at": "2025-01-08T10:30:00Z",
  "metadata": {
    "chapter": 3,
    "tags": ["ros2", "fundamentals"]
  }
}
```