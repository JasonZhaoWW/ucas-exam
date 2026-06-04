import sqlite3
from pathlib import Path

import chromadb

from src.api.database import get_chromadb_client, get_sqlite_connection, init_db


class TestInitDB:
    def test_creates_sqlite_tables(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        init_db(sqlite_path=str(db_path), chromadb_path=str(tmp_path / "chromadb"))

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "knowledge_bases" in tables
        assert "documents" in tables

    def test_knowledge_bases_table_schema(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        init_db(sqlite_path=str(db_path), chromadb_path=str(tmp_path / "chromadb"))

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("PRAGMA table_info(knowledge_bases)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        assert "id" in columns
        assert "name" in columns
        assert "description" in columns
        assert "created_at" in columns
        assert "updated_at" in columns

    def test_documents_table_schema(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        init_db(sqlite_path=str(db_path), chromadb_path=str(tmp_path / "chromadb"))

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("PRAGMA table_info(documents)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        assert "id" in columns
        assert "kb_id" in columns
        assert "filename" in columns
        assert "created_at" in columns

    def test_get_sqlite_connection_returns_connection(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        init_db(sqlite_path=str(db_path), chromadb_path=str(tmp_path / "chromadb"))

        conn = get_sqlite_connection()
        assert isinstance(conn, sqlite3.Connection)
        conn.close()

    def test_get_chromadb_client_returns_client(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        init_db(sqlite_path=str(db_path), chromadb_path=str(tmp_path / "chromadb"))

        client = get_chromadb_client()
        assert isinstance(client, chromadb.ClientAPI)
