from fastapi import FastAPI

from src.api.routes.documents import router as doc_router
from src.api.routes.knowledge_bases import router as kb_router

app = FastAPI(title="Knowledge Base API")
app.include_router(kb_router)
app.include_router(doc_router)
