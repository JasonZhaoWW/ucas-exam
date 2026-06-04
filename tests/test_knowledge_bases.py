from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.database import get_sqlite_connection, init_db
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


class TestCreateKnowledgeBase:
    def test_create_returns_201_with_kb_data(self, client):
        response = client.post(
            "/knowledge-bases",
            json={"name": "My KB", "description": "A test knowledge base"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "My KB"
        assert data["description"] == "A test knowledge base"
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    def test_create_with_empty_description_defaults_to_empty(self, client):
        response = client.post(
            "/knowledge-bases",
            json={"name": "Minimal KB"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Minimal KB"
        assert data["description"] == ""

    def test_create_without_name_returns_422(self, client):
        response = client.post(
            "/knowledge-bases",
            json={"description": "No name"},
        )

        assert response.status_code == 422


class TestListKnowledgeBases:
    def test_list_empty_returns_empty_list(self, client):
        response = client.get("/knowledge-bases")

        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_returns_all_kbs(self, client):
        client.post("/knowledge-bases", json={"name": "KB 1"})
        client.post("/knowledge-bases", json={"name": "KB 2"})
        client.post("/knowledge-bases", json={"name": "KB 3"})

        response = client.get("/knowledge-bases")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["items"]) == 3

    def test_list_respects_page_size(self, client):
        for i in range(5):
            client.post("/knowledge-bases", json={"name": f"KB {i}"})

        response = client.get("/knowledge-bases", params={"page": 1, "size": 2})

        data = response.json()
        assert len(data["items"]) == 2
        assert data["total"] == 5

    def test_list_page_2_returns_next_page(self, client):
        for i in range(5):
            client.post("/knowledge-bases", json={"name": f"KB {i}"})

        response = client.get("/knowledge-bases", params={"page": 2, "size": 2})

        data = response.json()
        assert len(data["items"]) == 2

    def test_list_defaults_to_page_1_size_10(self, client):
        for i in range(15):
            client.post("/knowledge-bases", json={"name": f"KB {i}"})

        response = client.get("/knowledge-bases")

        data = response.json()
        assert len(data["items"]) == 10
        assert data["total"] == 15


class TestGetKnowledgeBase:
    def test_get_existing_kb(self, client):
        create_resp = client.post(
            "/knowledge-bases", json={"name": "My KB", "description": "desc"}
        )
        kb_id = create_resp.json()["id"]

        response = client.get(f"/knowledge-bases/{kb_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == kb_id
        assert data["name"] == "My KB"
        assert data["description"] == "desc"
        assert data["document_count"] == 0

    def test_get_nonexistent_kb_returns_404(self, client):
        response = client.get("/knowledge-bases/nonexistent-id")

        assert response.status_code == 404
        assert response.json()["detail"] == "Knowledge base not found"

    def test_get_kb_reflects_document_count(self, client):
        create_resp = client.post(
            "/knowledge-bases", json={"name": "My KB"}
        )
        kb_id = create_resp.json()["id"]

        import uuid

        conn = get_sqlite_connection()
        now = "2024-01-01T00:00:00"
        for i in range(3):
            conn.execute(
                "INSERT INTO documents (id, kb_id, filename, created_at) "
                "VALUES (?, ?, ?, ?)",
                (str(uuid.uuid4()), kb_id, f"doc{i}.txt", now),
            )
        conn.commit()

        response = client.get(f"/knowledge-bases/{kb_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["document_count"] == 3


class TestUpdateKnowledgeBase:
    def test_update_name_and_description(self, client):
        create_resp = client.post(
            "/knowledge-bases", json={"name": "Old Name", "description": "Old desc"}
        )
        kb_id = create_resp.json()["id"]

        response = client.put(
            f"/knowledge-bases/{kb_id}",
            json={"name": "New Name", "description": "New desc"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "New Name"
        assert data["description"] == "New desc"
        assert data["id"] == kb_id

    def test_update_updates_updated_at(self, client):
        create_resp = client.post(
            "/knowledge-bases", json={"name": "My KB"}
        )
        kb_id = create_resp.json()["id"]
        original_updated_at = create_resp.json()["updated_at"]

        response = client.put(
            f"/knowledge-bases/{kb_id}",
            json={"name": "Updated KB"},
        )

        data = response.json()
        assert data["updated_at"] != original_updated_at

    def test_update_nonexistent_kb_returns_404(self, client):
        response = client.put(
            "/knowledge-bases/nonexistent-id",
            json={"name": "No KB"},
        )

        assert response.status_code == 404


class TestDeleteKnowledgeBase:
    def test_delete_existing_kb_returns_204(self, client):
        create_resp = client.post(
            "/knowledge-bases", json={"name": "To Delete"}
        )
        kb_id = create_resp.json()["id"]

        response = client.delete(f"/knowledge-bases/{kb_id}")

        assert response.status_code == 204

    def test_delete_removes_kb(self, client):
        create_resp = client.post(
            "/knowledge-bases", json={"name": "To Delete"}
        )
        kb_id = create_resp.json()["id"]

        client.delete(f"/knowledge-bases/{kb_id}")

        response = client.get(f"/knowledge-bases/{kb_id}")
        assert response.status_code == 404

    def test_delete_nonexistent_kb_returns_404(self, client):
        response = client.delete("/knowledge-bases/nonexistent-id")

        assert response.status_code == 404

    def test_delete_cascades_to_documents(self, client):
        create_resp = client.post(
            "/knowledge-bases", json={"name": "KB with docs"}
        )
        kb_id = create_resp.json()["id"]

        import uuid
        conn = get_sqlite_connection()
        now = "2024-01-01T00:00:00"
        doc_ids = []
        for i in range(3):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            conn.execute(
                "INSERT INTO documents (id, kb_id, filename, created_at) "
                "VALUES (?, ?, ?, ?)",
                (doc_id, kb_id, f"doc{i}.txt", now),
            )
        conn.commit()

        response = client.delete(f"/knowledge-bases/{kb_id}")

        assert response.status_code == 204
        for doc_id in doc_ids:
            cursor = conn.execute(
                "SELECT id FROM documents WHERE id = ?", (doc_id,)
            )
            assert cursor.fetchone() is None
