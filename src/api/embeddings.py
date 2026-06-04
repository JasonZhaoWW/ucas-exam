from __future__ import annotations

import os

from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None

DEFAULT_MODEL_NAME = "BAAI/bge-m3"


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        model_name = os.environ.get("KB_EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
        _model = SentenceTransformer(model_name)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(texts)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    model = _get_model()
    embedding = model.encode([query])
    return embedding[0].tolist()
