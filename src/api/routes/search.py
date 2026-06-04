from fastapi import APIRouter, HTTPException

from src.api.database import get_chromadb_client, get_sqlite_connection
from src.api.embeddings import embed_query
from src.api.models import SearchRequest, SearchResult

router = APIRouter(
    prefix="/knowledge-bases/{kb_id}/search",
    tags=["search"],
)


@router.post("", response_model=list[SearchResult])
def search_knowledge_base(kb_id: str, body: SearchRequest):
    conn = get_sqlite_connection()
    cursor = conn.execute(
        "SELECT id FROM knowledge_bases WHERE id = ?", (kb_id,)
    )
    if cursor.fetchone() is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    query_embedding = embed_query(body.query)

    chroma = get_chromadb_client()
    collection = chroma.get_collection("documents")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=body.top_k,
        where={"kb_id": kb_id},
        include=["documents", "distances", "metadatas"],
    )

    search_results = []
    if results["ids"] and results["ids"][0]:
        doc_ids = [m["document_id"] for m in results["metadatas"][0]]
        filenames = {}
        for doc_id in set(doc_ids):
            cursor = conn.execute(
                "SELECT id, filename FROM documents WHERE id = ?", (doc_id,)
            )
            row = cursor.fetchone()
            if row:
                filenames[row[0]] = row[1]

        for i, chunk_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            doc_id = metadata["document_id"]
            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=results["documents"][0][i],
                    score=results["distances"][0][i],
                    document_id=doc_id,
                    filename=filenames.get(doc_id, "unknown"),
                )
            )

    return search_results
