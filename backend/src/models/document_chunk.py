from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid

class DocumentChunk(BaseModel):
    id: str = str(uuid.uuid4())
    document_id: str
    content: str
    chunk_index: int
    vector_id: str
    metadata: Dict[str, Any] = {}