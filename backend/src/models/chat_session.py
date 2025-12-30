from pydantic import BaseModel
from datetime import datetime
import uuid
from typing import Dict, Any, Optional

class ChatSession(BaseModel):
    id: str = str(uuid.uuid4())
    start_time: datetime = datetime.now()
    last_activity: datetime = datetime.now()
    user_context: Dict[str, Any] = {}
    expires_at: datetime