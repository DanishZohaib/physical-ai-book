from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from datetime import datetime

from ..services.rag_service import RAGService
from ..services.postgres_service import PostgresService
from ..models.question import Question
from ..models.response import Response
from ..utils.logging import log_api_call, log_error, log_info, log_warning

router = APIRouter()

@router.post("/")
async def chat_endpoint(request: Dict[str, Any]):
    """
    Submit a question and receive an AI-generated response based on book content
    """
    log_info(f"Processing chat request with keys: {list(request.keys())}")
    try:
        # Validate required fields
        if "question" not in request:
            raise HTTPException(status_code=400, detail="Missing required field: question")

        # Extract parameters
        question_text = request["question"]
        page_context = request.get("page_context", "")
        selected_text = request.get("selected_text", None)
        session_id = request.get("session_id", "")

        # Validate question length
        if len(question_text.strip()) < 5:
            log_warning(f"Question too short: '{question_text[:20]}...' (length: {len(question_text)})")
            raise HTTPException(status_code=400, detail="Question must be at least 5 characters long")

        log_info(f"Processing question for session {session_id}: '{question_text[:50]}...'{'...' if len(question_text) > 50 else ''}")

        # Initialize services
        rag_service = RAGService()
        await rag_service.initialize_services()

        postgres_service = PostgresService()
        await postgres_service.connect()

        # Create question object
        question = Question(
            content=question_text,
            page_context=page_context,
            selected_text=selected_text,
            session_id=session_id
        )

        # Store question in database
        question_id = await postgres_service.store_question(question.dict())
        question.id = question_id  # Update with the stored ID

        # Generate response using RAG service
        response = await rag_service.answer_question(question)

        # Store response in database
        response_id = await postgres_service.store_response(response.dict())
        response.id = response_id  # Update with the stored ID

        # Prepare the response
        result = {
            "id": response.id,
            "question_id": response.question_id,
            "answer": response.content,
            "sources": response.sources,
            "confidence": response.confidence_score,
            "session_id": response.session_id,
            "timestamp": datetime.now().isoformat()
        }

        log_api_call("/chat", "POST", {"session_id": session_id, "has_selected_text": bool(selected_text)}, 200)
        log_info(f"Successfully processed chat request, response length: {len(response.content)}")
        return result

    except HTTPException as he:
        log_error(he, f"chat_endpoint - HTTP {he.status_code}")
        raise
    except Exception as e:
        log_error(e, "chat_endpoint")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")


@router.post("/start-session")
async def start_session(request: Dict[str, Any]):
    """
    Start a new chat session
    """
    log_info(f"Creating new chat session with context: {request.get('initial_context', {})}")
    try:
        # Extract parameters
        initial_context = request.get("initial_context", {})
        page = initial_context.get("page", "")
        user_agent = initial_context.get("user_agent", "")

        from ..models.chat_session import ChatSession
        from datetime import timedelta
        import uuid

        # Create session object with 1-hour expiration
        session_id = str(uuid.uuid4())
        start_time = datetime.now()
        expires_at = start_time + timedelta(hours=1)

        session = ChatSession(
            id=session_id,
            start_time=start_time,
            last_activity=start_time,
            user_context={
                "page": page,
                "user_agent": user_agent
            },
            expires_at=expires_at
        )

        # Initialize services
        postgres_service = PostgresService()
        await postgres_service.connect()

        # Store session in database
        session_id = await postgres_service.store_chat_session(session.dict())

        log_api_call("/chat/start-session", "POST", {"page": page, "user_agent": user_agent}, 200)
        log_info(f"Successfully created session: {session_id[:8]}...")

        return {
            "session_id": session_id,
            "created_at": start_time.isoformat(),
            "expires_at": expires_at.isoformat()
        }

    except Exception as e:
        log_error(e, "start_session")
        raise HTTPException(status_code=500, detail=f"Error starting session: {str(e)}")