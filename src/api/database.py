import sqlite3
from pathlib import Path
from typing import Any

import chromadb

_sqlite_conn: sqlite3.Connection | None = None
_chromadb_client: Any = None


def init_db(
    sqlite_path: str = "data/knowledge_base.db",
    chromadb_path: str = "data/chromadb",
) -> None:
    global _sqlite_conn, _chromadb_client

    Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    Path(chromadb_path).mkdir(parents=True, exist_ok=True)

    _sqlite_conn = sqlite3.connect(sqlite_path, check_same_thread=False)
    _sqlite_conn.execute("PRAGMA journal_mode=WAL")
    _sqlite_conn.execute("PRAGMA foreign_keys=ON")

    _sqlite_conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_bases (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    _sqlite_conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            kb_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
        )
    """)

    _sqlite_conn.commit()

    _chromadb_client = chromadb.PersistentClient(path=chromadb_path)


def get_sqlite_connection() -> sqlite3.Connection:
    if _sqlite_conn is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _sqlite_conn


def get_chromadb_client() -> Any:
    if _chromadb_client is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _chromadb_client
