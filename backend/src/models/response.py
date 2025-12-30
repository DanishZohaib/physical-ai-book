from pydantic import BaseModel
from datetime import datetime
import uuid
from typing import List, Dict, Any, Optional

class Response(BaseModel):
    id: str = str(uuid.uuid4())
    question_id: str
    content: str
    sources: List[Dict[str, Any]] = []
    confidence_score: float
    timestamp: datetime = datetime.now()
    session_id: str