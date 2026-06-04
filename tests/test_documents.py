from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.database import get_chromadb_client, get_sqlite_connection, init_db
from src.api.main import app


@pytest.fixture(autouse=True)
def setup_db(tmp_path: Path):
    init_db(
        sqlite_path=str(tmp_path / "test.db"),
        chromadb_path=str(tmp_path / "chromadb"),
    )


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def kb_id(client):
    resp = client.post("/knowledge-bases", json={"name": "Test KB"})
    return resp.json()["id"]


class TestUploadDocumentRawText:
    @patch("src.api.routes.documents.embed_texts")
    def test_upload_raw_text_returns_201(self, mock_embed, client, kb_id):
        mock_embed.return_value = [[0.1] * 10]

        response = client.post(
            f"/knowledge-bases/{kb_id}/documents",
            data={"text": "这是测试文本。"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["kb_id"] == kb_id
        assert data["filename"] == "raw_text.txt"
        assert "id" in data
        assert "created_at" in data

    @patch("src.api.routes.documents.embed_texts")
    def test_upload_raw_text_stores_in_sqlite(self, mock_embed, client, kb_id):
        mock_embed.return_value = [[0.1] * 10]

        response = client.post(
            f"/knowledge-bases/{kb_id}/documents",
            data={"text": "这是测试文本。"},
        )
        doc_id = response.json()["id"]

        conn = get_sqlite_connection()
        cursor = conn.execute(
            "SELECT id, kb_id, filename FROM documents WHERE id = ?", (doc_id,)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[1] == kb_id
        assert row[2] == "raw_text.txt"

    @patch("src.api.routes.documents.embed_texts")
    def test_upload_raw_text_stores_chunks_in_chromadb(self, mock_embed, client, kb_id):
        mock_embed.side_effect = lambda texts: [[0.1] * 10 for _ in texts]

        long_text = "这是一段很长的文本。" * 100
        response = client.post(
            f"/knowledge-bases/{kb_id}/documents",
            data={"text": long_text},
        )
        doc_id = response.json()["id"]

        chroma = get_chromadb_client()
        collection = chroma.get_collection("documents")
        results = collection.get(where={"document_id": doc_id})
        assert len(results["ids"]) >= 1
        mock_embed.assert_called_once()


class TestUploadDocumentTxtFile:
    @patch("src.api.routes.documents.embed_texts")
    def test_upload_txt_file_returns_201(self, mock_embed, client, kb_id):
        mock_embed.side_effect = lambda texts: [[0.1] * 10 for _ in texts]

        response = client.post(
            f"/knowledge-bases/{kb_id}/documents",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["kb_id"] == kb_id
        assert data["filename"] == "test.txt"

    @patch("src.api.routes.documents.embed_texts")
    def test_upload_txt_file_stores_in_sqlite(self, mock_embed, client, kb_id):
        mock_embed.side_effect = lambda texts: [[0.1] * 10 for _ in texts]

        response = client.post(
            f"/knowledge-bases/{kb_id}/documents",
            files={"file": ("doc.txt", b"some content", "text/plain")},
        )
        doc_id = response.json()["id"]

        conn = get_sqlite_connection()
        cursor = conn.execute(
            "SELECT id, kb_id, filename FROM documents WHERE id = ?", (doc_id,)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[2] == "doc.txt"


class TestUploadDocumentMdFile:
    @patch("src.api.routes.documents.embed_texts")
    def test_upload_md_file_returns_201(self, mock_embed, client, kb_id):
        mock_embed.side_effect = lambda texts: [[0.1] * 10 for _ in texts]

        response = client.post(
            f"/knowledge-bases/{kb_id}/documents",
            files={"file": ("readme.md", b"# Hello\n\nWorld", "text/markdown")},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["kb_id"] == kb_id
        assert data["filename"] == "readme.md"


class TestUploadDocumentNotFound:
    def test_upload_to_nonexistent_kb_returns_404(self, client):
        response = client.post(
            "/knowledge-bases/nonexistent-id/documents",
            data={"text": "hello"},
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "Knowledge base not found"


class TestListDocuments:
    def test_list_empty_returns_empty_list(self, client, kb_id):
        response = client.get(f"/knowledge-bases/{kb_id}/documents")

        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0

    @patch("src.api.routes.documents.embed_texts")
    def test_list_returns_all_documents(self, mock_embed, client, kb_id):
        mock_embed.side_effect = lambda texts: [[0.1] * 10 for _ in texts]

        for i in range(3):
            client.post(
                f"/knowledge-bases/{kb_id}/documents",
                data={"text": f"doc {i}"},
            )

        response = client.get(f"/knowledge-bases/{kb_id}/documents")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["items"]) == 3

    @patch("src.api.routes.documents.embed_texts")
    def test_list_respects_page_size(self, mock_embed, client, kb_id):
        mock_embed.side_effect = lambda texts: [[0.1] * 10 for _ in texts]

        for i in range(5):
            client.post(
                f"/knowledge-bases/{kb_id}/documents",
                data={"text": f"doc {i}"},
            )

        response = client.get(
            f"/knowledge-bases/{kb_id}/documents",
            params={"page": 1, "size": 2},
        )

        data = response.json()
        assert len(data["items"]) == 2
        assert data["total"] == 5

    def test_list_nonexistent_kb_returns_404(self, client):
        response = client.get("/knowledge-bases/nonexistent-id/documents")

        assert response.status_code == 404


class TestDeleteDocument:
    @patch("src.api.routes.documents.embed_texts")
    def test_delete_returns_204(self, mock_embed, client, kb_id):
        mock_embed.side_effect = lambda texts: [[0.1] * 10 for _ in texts]

        create_resp = client.post(
            f"/knowledge-bases/{kb_id}/documents",
            data={"text": "to delete"},
        )
        doc_id = create_resp.json()["id"]

        response = client.delete(
            f"/knowledge-bases/{kb_id}/documents/{doc_id}"
        )

        assert response.status_code == 204

    @patch("src.api.routes.documents.embed_texts")
    def test_delete_removes_from_sqlite(self, mock_embed, client, kb_id):
        mock_embed.side_effect = lambda texts: [[0.1] * 10 for _ in texts]

        create_resp = client.post(
            f"/knowledge-bases/{kb_id}/documents",
            data={"text": "to delete"},
        )
        doc_id = create_resp.json()["id"]

        client.delete(f"/knowledge-bases/{kb_id}/documents/{doc_id}")

        conn = get_sqlite_connection()
        cursor = conn.execute(
            "SELECT id FROM documents WHERE id = ?", (doc_id,)
        )
        assert cursor.fetchone() is None

    @patch("src.api.routes.documents.embed_texts")
    def test_delete_removes_chunks_from_chromadb(self, mock_embed, client, kb_id):
        mock_embed.side_effect = lambda texts: [[0.1] * 10 for _ in texts]

        create_resp = client.post(
            f"/knowledge-bases/{kb_id}/documents",
            data={"text": "to delete"},
        )
        doc_id = create_resp.json()["id"]

        client.delete(f"/knowledge-bases/{kb_id}/documents/{doc_id}")

        chroma = get_chromadb_client()
        collection = chroma.get_collection("documents")
        results = collection.get(where={"document_id": doc_id})
        assert len(results["ids"]) == 0

    def test_delete_nonexistent_doc_returns_404(self, client, kb_id):
        response = client.delete(
            f"/knowledge-bases/{kb_id}/documents/nonexistent-id"
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "Document not found"
