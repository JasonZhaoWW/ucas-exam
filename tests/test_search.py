from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.database import init_db
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


class TestSearchKnowledgeBaseNotFound:
    def test_search_nonexistent_kb_returns_404(self, client):
        response = client.post(
            "/knowledge-bases/nonexistent-id/search",
            json={"query": "test"},
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "Knowledge base not found"


class TestSearchValidation:
    def test_empty_query_returns_422(self, client, kb_id):
        response = client.post(
            f"/knowledge-bases/{kb_id}/search",
            json={"query": ""},
        )

        assert response.status_code == 422


class TestSearchBasic:
    @patch("src.api.routes.search.embed_query")
    @patch("src.api.routes.documents.embed_texts")
    def test_search_returns_results_with_correct_fields(
        self, mock_embed_texts, mock_embed_query, client, kb_id
    ):
        mock_embed_texts.side_effect = lambda texts: [[0.1] * 10 for _ in texts]
        mock_embed_query.return_value = [0.1] * 10

        client.post(
            f"/knowledge-bases/{kb_id}/documents",
            data={"text": "春天来了。万物复苏。"},
        )

        response = client.post(
            f"/knowledge-bases/{kb_id}/search",
            json={"query": "春天"},
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

        result = data[0]
        assert "chunk_id" in result
        assert "text" in result
        assert "score" in result
        assert "document_id" in result
        assert "filename" in result


class TestSearchTopK:
    @patch("src.api.routes.search.embed_query")
    @patch("src.api.routes.documents.embed_texts")
    def test_top_k_limits_results(
        self, mock_embed_texts, mock_embed_query, client, kb_id
    ):
        mock_embed_texts.side_effect = lambda texts: [[0.1] * 10 for _ in texts]
        mock_embed_query.return_value = [0.1] * 10

        client.post(
            f"/knowledge-bases/{kb_id}/documents",
            data={"text": "第一句。第二句。第三句。第四句。第五句。第六句。"},
        )

        response = client.post(
            f"/knowledge-bases/{kb_id}/search",
            json={"query": "句子", "top_k": 2},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 2

    @patch("src.api.routes.search.embed_query")
    @patch("src.api.routes.documents.embed_texts")
    def test_default_top_k_is_5(
        self, mock_embed_texts, mock_embed_query, client, kb_id
    ):
        mock_embed_texts.side_effect = lambda texts: [[0.1] * 10 for _ in texts]
        mock_embed_query.return_value = [0.1] * 10

        long_text = "。".join([f"第{i}段" for i in range(10)]) + "。"
        client.post(
            f"/knowledge-bases/{kb_id}/documents",
            data={"text": long_text},
        )

        response = client.post(
            f"/knowledge-bases/{kb_id}/search",
            json={"query": "段落"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 5


class TestSearchFilteredByKB:
    @patch("src.api.routes.search.embed_query")
    @patch("src.api.routes.documents.embed_texts")
    def test_search_only_returns_results_from_specified_kb(
        self, mock_embed_texts, mock_embed_query, client
    ):
        mock_embed_texts.side_effect = lambda texts: [[0.1] * 10 for _ in texts]
        mock_embed_query.return_value = [0.1] * 10

        kb1 = client.post("/knowledge-bases", json={"name": "KB1"}).json()["id"]
        kb2 = client.post("/knowledge-bases", json={"name": "KB2"}).json()["id"]

        doc1_resp = client.post(
            f"/knowledge-bases/{kb1}/documents",
            data={"text": "故乡的风景。"},
        )
        doc1_id = doc1_resp.json()["id"]

        client.post(
            f"/knowledge-bases/{kb2}/documents",
            data={"text": "春天的花。"},
        )

        response = client.post(
            f"/knowledge-bases/{kb1}/search",
            json={"query": "风景"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        for result in data:
            assert result["document_id"] == doc1_id


class TestSearchScoreOrder:
    @patch("src.api.routes.search.embed_query")
    @patch("src.api.routes.documents.embed_texts")
    def test_results_ordered_by_score(
        self, mock_embed_texts, mock_embed_query, client, kb_id
    ):
        mock_embed_texts.side_effect = lambda texts: [[0.1] * 10 for _ in texts]
        mock_embed_query.return_value = [0.1] * 10

        long1 = "春天来了" * 130 + "。"
        long2 = "夏天到了" * 130 + "。"
        client.post(
            f"/knowledge-bases/{kb_id}/documents",
            data={"text": long1 + long2},
        )

        response = client.post(
            f"/knowledge-bases/{kb_id}/search",
            json={"query": "季节", "top_k": 10},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2

        scores = [r["score"] for r in data]
        assert scores == sorted(scores)
