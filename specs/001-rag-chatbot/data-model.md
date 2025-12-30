# Data Model: RAG Chatbot for Physical AI Book

## Entities

### Question
- **id**: string (UUID) - Unique identifier for the question
- **content**: string - The natural language query from the user
- **page_context**: string - The current page/chapter context where the question was asked
- **selected_text**: string (optional) - Specific text selected by the user when asking the question
- **timestamp**: datetime - When the question was submitted
- **session_id**: string - Reference to the chat session

### Document
- **id**: string (UUID) - Unique identifier for the document
- **title**: string - Title of the document/chapter
- **content**: string - Full text content of the document
- **source_path**: string - Path to the original document file
- **chunk_count**: integer - Number of text chunks this document is divided into
- **created_at**: datetime - When the document was indexed
- **updated_at**: datetime - When the document was last updated

### DocumentChunk
- **id**: string (UUID) - Unique identifier for the chunk
- **document_id**: string - Reference to the parent document
- **content**: string - Text content of this chunk
- **chunk_index**: integer - Position of this chunk within the document
- **vector_id**: string - Reference to the vector in Qdrant
- **metadata**: object - Additional metadata for search and retrieval

### ChatSession
- **id**: string (UUID) - Unique identifier for the session
- **start_time**: datetime - When the session started
- **last_activity**: datetime - When the last message was exchanged
- **user_context**: object - Information about the user's current location in the book
- **expires_at**: datetime - When the session expires (1 hour after last activity)

### Response
- **id**: string (UUID) - Unique identifier for the response
- **question_id**: string - Reference to the associated question
- **content**: string - The chatbot's response to the question
- **sources**: array of strings - Document IDs that contributed to the response
- **confidence_score**: float - Confidence level of the response (0.0-1.0)
- **timestamp**: datetime - When the response was generated
- **session_id**: string - Reference to the chat session

### UserMessage
- **id**: string (UUID) - Unique identifier for the message
- **session_id**: string - Reference to the chat session
- **role**: string - Either "user" or "assistant"
- **content**: string - The message content
- **timestamp**: datetime - When the message was sent
- **metadata**: object - Additional context (page, selected text, etc.)

## Relationships

- ChatSession 1 --- * UserMessage (one session to many messages)
- ChatSession 1 --- * Question (one session to many questions)
- Question 1 --- 1 Response (one question to one response)
- Document 1 --- * DocumentChunk (one document to many chunks)
- Response 1 --- * Document (one response to many source documents)

## Validation Rules

- Question content must be between 5 and 1000 characters
- Document chunks should be between 200-1000 tokens for optimal retrieval
- Chat sessions must expire after 1 hour of inactivity
- Responses must include at least one source document reference
- Confidence scores must be between 0.0 and 1.0

## State Transitions

- ChatSession: active → expired (after 1 hour of inactivity)
- Question: pending → processing → completed → answered