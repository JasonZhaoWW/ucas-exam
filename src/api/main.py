from fastapi import FastAPI

from src.api.routes.knowledge_bases import router as kb_router

app = FastAPI(title="Knowledge Base API")
app.include_router(kb_router)
