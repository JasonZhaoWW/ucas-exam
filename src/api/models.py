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


class Document(BaseModel):
    id: str
    kb_id: str
    filename: str
    created_at: datetime = Field(default_factory=_utcnow)
