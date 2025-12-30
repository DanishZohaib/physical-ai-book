from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import uuid

class Question(BaseModel):
    id: str = str(uuid.uuid4())
    content: str
    page_context: str
    selected_text: Optional[str] = None
    timestamp: datetime = datetime.now()
    session_id: str