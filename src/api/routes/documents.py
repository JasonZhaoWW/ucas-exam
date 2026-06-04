import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query
from fastapi.datastructures import UploadFile

from src.api.chunking import chunk_text
from src.api.database import get_chromadb_client, get_sqlite_connection
from src.api.embeddings import embed_texts
from src.api.models import DocumentList, DocumentResponse

router = APIRouter(
    prefix="/knowledge-bases/{kb_id}/documents",
    tags=["documents"],
)


@router.post("", status_code=201, response_model=DocumentResponse)
async def upload_document(
    kb_id: str,
    file: Annotated[UploadFile | None, File()] = None,
    text: Annotated[str | None, Form()] = None,
):
    conn = get_sqlite_connection()
    cursor = conn.execute(
        "SELECT id FROM knowledge_bases WHERE id = ?", (kb_id,)
    )
    if cursor.fetchone() is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    if file is not None:
        content = await file.read()
        text_content = content.decode("utf-8")
        filename = file.filename or "uploaded_file"
    elif text is not None:
        text_content = text
        filename = "raw_text.txt"
    else:
        raise HTTPException(
            status_code=422, detail="Either 'file' or 'text' must be provided"
        )

    chunks = chunk_text(text_content)
    if not chunks:
        chunks = [text_content] if text_content else [""]

    embeddings = embed_texts(chunks)

    doc_id = str(uuid.uuid4())
    now = datetime.now(UTC)

    conn.execute(
        "INSERT INTO documents (id, kb_id, filename, created_at) "
        "VALUES (?, ?, ?, ?)",
        (doc_id, kb_id, filename, now.isoformat()),
    )
    conn.commit()

    chroma = get_chromadb_client()
    collection = chroma.get_or_create_collection("documents")
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"document_id": doc_id, "kb_id": kb_id}] * len(chunks)
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return DocumentResponse(
        id=doc_id,
        kb_id=kb_id,
        filename=filename,
        created_at=now,
    )


@router.get("", response_model=DocumentList)
def list_documents(
    kb_id: str,
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
):
    conn = get_sqlite_connection()
    cursor = conn.execute(
        "SELECT id FROM knowledge_bases WHERE id = ?", (kb_id,)
    )
    if cursor.fetchone() is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    offset = (page - 1) * size

    cursor = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE kb_id = ?", (kb_id,)
    )
    total = cursor.fetchone()[0]

    cursor = conn.execute(
        "SELECT id, kb_id, filename, created_at "
        "FROM documents WHERE kb_id = ? "
        "ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (kb_id, size, offset),
    )
    items = [
        DocumentResponse(
            id=row[0],
            kb_id=row[1],
            filename=row[2],
            created_at=row[3],
        )
        for row in cursor.fetchall()
    ]

    return DocumentList(items=items, total=total)


@router.delete("/{doc_id}", status_code=204)
def delete_document(kb_id: str, doc_id: str):
    conn = get_sqlite_connection()
    cursor = conn.execute(
        "SELECT id FROM documents WHERE id = ? AND kb_id = ?",
        (doc_id, kb_id),
    )
    if cursor.fetchone() is None:
        raise HTTPException(status_code=404, detail="Document not found")

    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()

    chroma = get_chromadb_client()
    collection = chroma.get_collection("documents")
    results = collection.get(where={"document_id": doc_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])
