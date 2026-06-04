from datetime import UTC, datetime

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


class KnowledgeBase(BaseModel):
    id: str
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class KnowledgeBaseCreate(BaseModel):
    name: str
    description: str = ""


class KnowledgeBaseResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime


class KnowledgeBaseList(BaseModel):
    items: list[KnowledgeBaseResponse]
    total: int


class KnowledgeBaseDetail(KnowledgeBaseResponse):
    document_count: int


class KnowledgeBaseUpdate(BaseModel):
    name: str
    description: str = ""


class Document(BaseModel):
    id: str
    kb_id: str
    filename: str
    created_at: datetime = Field(default_factory=_utcnow)


class DocumentResponse(BaseModel):
    id: str
    kb_id: str
    filename: str
    created_at: datetime


class DocumentList(BaseModel):
    items: list[DocumentResponse]
    total: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = 5


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    document_id: str
    filename: str
