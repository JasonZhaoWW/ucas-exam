import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query

from src.api.database import get_sqlite_connection
from src.api.models import (
    KnowledgeBaseCreate,
    KnowledgeBaseDetail,
    KnowledgeBaseList,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
)

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


@router.post("", status_code=201, response_model=KnowledgeBaseResponse)
def create_knowledge_base(body: KnowledgeBaseCreate):
    now = datetime.now(UTC).isoformat()
    kb_id = str(uuid.uuid4())
    conn = get_sqlite_connection()
    conn.execute(
        "INSERT INTO knowledge_bases "
        "(id, name, description, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (kb_id, body.name, body.description, now, now),
    )
    conn.commit()
    return KnowledgeBaseResponse(
        id=kb_id,
        name=body.name,
        description=body.description,
        created_at=now,
        updated_at=now,
    )


@router.get("", response_model=KnowledgeBaseList)
def list_knowledge_bases(
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
):
    conn = get_sqlite_connection()
    offset = (page - 1) * size

    cursor = conn.execute("SELECT COUNT(*) FROM knowledge_bases")
    total = cursor.fetchone()[0]

    cursor = conn.execute(
        "SELECT id, name, description, created_at, updated_at "
        "FROM knowledge_bases "
        "ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (size, offset),
    )
    items = [
        KnowledgeBaseResponse(
            id=row[0],
            name=row[1],
            description=row[2],
            created_at=row[3],
            updated_at=row[4],
        )
        for row in cursor.fetchall()
    ]

    return KnowledgeBaseList(items=items, total=total)


@router.get("/{kb_id}", response_model=KnowledgeBaseDetail)
def get_knowledge_base(kb_id: str):
    conn = get_sqlite_connection()
    cursor = conn.execute(
        "SELECT id, name, description, created_at, updated_at "
        "FROM knowledge_bases WHERE id = ?",
        (kb_id,),
    )
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    cursor = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE kb_id = ?", (kb_id,)
    )
    document_count = cursor.fetchone()[0]

    return KnowledgeBaseDetail(
        id=row[0],
        name=row[1],
        description=row[2],
        created_at=row[3],
        updated_at=row[4],
        document_count=document_count,
    )


@router.put("/{kb_id}", response_model=KnowledgeBaseResponse)
def update_knowledge_base(kb_id: str, body: KnowledgeBaseUpdate):
    conn = get_sqlite_connection()
    cursor = conn.execute(
        "SELECT id FROM knowledge_bases WHERE id = ?", (kb_id,)
    )
    if cursor.fetchone() is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    now = datetime.now(UTC).isoformat()
    conn.execute(
        "UPDATE knowledge_bases "
        "SET name = ?, description = ?, updated_at = ? WHERE id = ?",
        (body.name, body.description, now, kb_id),
    )
    conn.commit()

    cursor = conn.execute(
        "SELECT id, name, description, created_at, updated_at "
        "FROM knowledge_bases WHERE id = ?",
        (kb_id,),
    )
    row = cursor.fetchone()
    return KnowledgeBaseResponse(
        id=row[0],
        name=row[1],
        description=row[2],
        created_at=row[3],
        updated_at=row[4],
    )


@router.delete("/{kb_id}", status_code=204)
def delete_knowledge_base(kb_id: str):
    conn = get_sqlite_connection()
    cursor = conn.execute(
        "SELECT id FROM knowledge_bases WHERE id = ?", (kb_id,)
    )
    if cursor.fetchone() is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    conn.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
    conn.commit()
