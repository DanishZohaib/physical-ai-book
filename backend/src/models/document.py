from pydantic import BaseModel
from datetime import datetime
import uuid
from typing import Optional

class Document(BaseModel):
    id: str = str(uuid.uuid4())
    title: str
    content: str
    source_path: str
    chunk_count: int
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()