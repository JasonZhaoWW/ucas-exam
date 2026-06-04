from fastapi import FastAPI

from src.api.routes.documents import router as doc_router
from src.api.routes.knowledge_bases import router as kb_router
from src.api.routes.search import router as search_router

app = FastAPI(title="Knowledge Base API")
app.include_router(kb_router)
app.include_router(doc_router)
app.include_router(search_router)
