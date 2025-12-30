from pydantic import BaseModel
from datetime import datetime
import uuid
from typing import Dict, Any, Optional

class MessageRole:
    USER = "user"
    ASSISTANT = "assistant"

class UserMessage(BaseModel):
    id: str = str(uuid.uuid4())
    session_id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = datetime.now()
    metadata: Dict[str, Any] = {}