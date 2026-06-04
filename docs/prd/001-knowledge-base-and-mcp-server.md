# PRD: Knowledge Base & MCP Server

## Problem Statement

As a user, I need a system that lets me upload text documents into knowledge bases, then search them using natural language queries with semantic understanding. The search results must be accessible to an AI agent via the Model Context Protocol (MCP), so the agent can answer questions like "帮我查一下春天相关内容" by retrieving relevant knowledge base content.

## Solution

A two-service architecture:
1. **Knowledge Base API** (FastAPI) — CRUD for knowledge bases and documents, semantic search with vector embeddings
2. **MCP Server** (stdio) — wraps all API features as MCP tools, callable by any MCP-compatible agent

The user uploads documents (.txt, .md, or raw text), the system chunks them into sentences, embeds them with BAAI/bge-m3, and stores them in ChromaDB. Queries are embedded with the same model and matched via cosine similarity.

## User Stories

1. As a user, I want to create a knowledge base with a name and description, so that I can organize my documents
2. As a user, I want to list all my knowledge bases with pagination, so that I can browse them
3. As a user, I want to view details of a specific knowledge base, so that I can see its metadata and document count
4. As a user, I want to update a knowledge base's name and description, so that I can keep metadata accurate
5. As a user, I want to delete a knowledge base and all its documents, so that I can clean up unused data
6. As a user, I want to upload a .txt file to a knowledge base, so that the system can index its content
7. As a user, I want to upload a .md file to a knowledge base, so that markdown documents are indexed
8. As a user, I want to paste raw text directly into a knowledge base, so that I can add content without creating a file
9. As a user, I want to list documents in a knowledge base with pagination, so that I can manage uploaded content
10. As a user, I want to delete a specific document from a knowledge base, so that I can remove outdated content
11. As a user, I want to search a knowledge base with a natural language query, so that I find semantically relevant content
12. As a user, I want search to work with Chinese semantics (e.g., "小孩子" finds "故乡"), so that language nuance is preserved
13. As a user, I want to specify `top_k` in search, so that I control how many results I get
14. As a user, I want clear error messages when my query is empty, so that I know what to fix
15. As a user, I want clear error messages when a knowledge base doesn't exist, so that I can correct my request
16. As a user, I want clear error messages when retrieval fails, so that I know something went wrong
17. As a user, I want the system to timeout gracefully if a tool call takes too long, so that the agent doesn't hang
18. As an agent developer, I want MCP tools for all knowledge base operations, so that my agent can manage KBs programmatically
19. As an agent developer, I want MCP tool descriptions to be clear, so that the agent knows when to call each tool
20. As a user, I want the API to auto-generate docs at /docs, so that I can explore endpoints interactively
21. As a developer, I want configuration via .env file, so that I can customize paths and ports without code changes
22. As a developer, I want basic unit tests, so that I can verify core logic works correctly

## Implementation Decisions

### Architecture

- **Two separate processes**: FastAPI server runs independently, MCP server connects to it via HTTP
- **MCP server reads `KB_API_URL`** from environment (default `http://localhost:8000`)
- **No auto-start**: user starts FastAPI first, then configures MCP server in their agent

### Knowledge Base API (FastAPI)

- Endpoints follow RESTful conventions with nested resources for documents
- Offset-based pagination: `?page=1&size=10`
- ChromaDB collection per knowledge base (one collection = one KB)
- SQLite stores knowledge base metadata (id, name, description, created_at, updated_at)
- Document metadata stored in SQLite (id, kb_id, filename, created_at)
- Chunks stored in ChromaDB with metadata linking back to document

### Text Chunking

- Sentence-based splitting on Chinese punctuation: `。！？\n`
- Merge short sentences up to max ~500 characters per chunk
- No overlap needed — sentences are complete semantic units

### Embeddings

- Model: `BAAI/bge-m3` via sentence-transformers
- ~2.2GB download on first run
- All chunks embedded at upload time (not at query time)
- Queries embedded at search time

### MCP Server

- Transport: stdio
- SDK: `mcp` Python SDK
- 9 tools wrapping all FastAPI endpoints:
  - `create_knowledge_base`, `list_knowledge_bases`, `get_knowledge_base`
  - `update_knowledge_base`, `delete_knowledge_base`
  - `upload_document`, `list_documents`, `delete_document`
  - `search_knowledge_base`
- Each tool has a clear description for agent discovery
- Error handling:
  - Empty query → immediate error, no API call
  - KB not found → catch 404, return descriptive error
  - Retrieval failure → catch exceptions, return generic error
  - Timeout → 30s timeout on HTTP calls, return timeout error

### Configuration

- `.env` file via `python-dotenv`
- Variables: `KB_API_URL`, `KB_CHROMADB_PATH`, `KB_SQLITE_PATH`, `KB_EMBEDDING_MODEL`, `KB_PORT`

### Dev Tooling

- Package manager: `uv`
- Linter: `ruff`
- Type checker: `basedpyright`
- Testing: `pytest`

### Project Structure

```
ucas-exam/
├── README.md
├── pyproject.toml
├── .env.example
├── src/
│   ├── api/                    # FastAPI knowledge base service
│   │   ├── main.py
│   │   ├── models.py           # Pydantic schemas
│   │   ├── database.py         # SQLite + ChromaDB init
│   │   ├── chunking.py         # Text chunking
│   │   ├── embeddings.py       # bge-m3 wrapper
│   │   └── routes/
│   │       ├── knowledge_bases.py
│   │       ├── documents.py
│   │       └── search.py
│   └── mcp/                    # MCP server
│       └── server.py
└── tests/
    ├── test_chunking.py
    ├── test_search.py
    └── test_error_handling.py
```

## Testing Decisions

- **What makes a good test**: tests should verify external behavior (API responses, search results), not implementation details (internal chunk representation, embedding vectors)
- **Modules to test**:
  - `chunking.py` — Chinese sentence splitting, max chunk size enforcement
  - `search` route — end-to-end search relevance ("小孩子" → "故乡")
  - Error handling — empty query, missing KB, invalid input
- **No mocking of embeddings in unit tests** — use real bge-m3 for relevance tests (slow first run, then cached)

## Out of Scope

- PDF or .docx file support
- User authentication or multi-tenancy
- Streaming SSE from FastAPI (streaming is the agent's responsibility)
- Production deployment (Docker, nginx, etc.)
- Web UI for knowledge base management
- Document update/edit (only create and delete)
- Cross-knowledge-base search

## Further Notes

- The "streaming" requirement (查询的返回需要流式返回) refers to the agent's response to the user, not the API. The MCP tool returns search results as a complete JSON response. The agent handles token-by-token streaming.
- The submission requires: code, README, running instructions, implementation explanation, and personal resume.
- Deadline: same day 18:00, email to jiangtai20@mails.ucas.ac.cn
