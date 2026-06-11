from typing import Optional, List
from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class ChatRequest(BaseModel):
    query:      str           = Field(..., min_length=1, max_length=2000)
    doc_filter: Optional[str] = Field(None)


class CitationModel(BaseModel):
    chunk_id:      str
    document_name: str
    page_number:   int
    score:         float
    excerpt:       str
    url:           Optional[str] = None
    source_type:   Optional[str] = "document"   # "document" | "web"


class ChatResponse(BaseModel):
    answer:      str
    sources:     List[CitationModel] = []
    confidence:  float               = 0.0
    query_type:  Optional[str]       = None
    search_mode: Optional[str]       = None   # WEB_ONLY | DOC_ONLY | HYBRID
    blocked:     bool                = False
    blocked_by:  Optional[str]       = None
    safety_note: str


class DocumentInfo(BaseModel):
    doc_id:      str
    filename:    str
    chunk_count: int
    status:      str = "indexed"


class UploadResponse(BaseModel):
    doc_id:       str
    filename:     str
    total_chunks: int
    message:      str


class DeleteResponse(BaseModel):
    doc_id:  str
    message: str


class HealthResponse(BaseModel):
    status:             str
    version:            str
    embedding_model:    str
    vector_store:       str
    llm_model:          str
    web_search:         str   # "tavily" | "scraping" | "unavailable"
    langsmith_enabled:  bool
