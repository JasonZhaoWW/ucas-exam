from datetime import datetime

import pytest
from pydantic import ValidationError

from src.api.models import Document, KnowledgeBase


class TestKnowledgeBase:
    def test_create_with_all_fields(self):
        kb = KnowledgeBase(
            id="kb-123",
            name="My KB",
            description="A test KB",
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
        )
        assert kb.id == "kb-123"
        assert kb.name == "My KB"
        assert kb.description == "A test KB"
        assert kb.created_at == datetime(2024, 1, 1)
        assert kb.updated_at == datetime(2024, 1, 1)

    def test_create_with_defaults(self):
        kb = KnowledgeBase(id="kb-123", name="My KB")
        assert kb.description == ""
        assert isinstance(kb.created_at, datetime)
        assert isinstance(kb.updated_at, datetime)

    def test_missing_id_raises_error(self):
        with pytest.raises(ValidationError):
            KnowledgeBase(name="My KB")

    def test_missing_name_raises_error(self):
        with pytest.raises(ValidationError):
            KnowledgeBase(id="kb-123")


class TestDocument:
    def test_create_with_all_fields(self):
        doc = Document(
            id="doc-456",
            kb_id="kb-123",
            filename="test.txt",
            created_at=datetime(2024, 1, 1),
        )
        assert doc.id == "doc-456"
        assert doc.kb_id == "kb-123"
        assert doc.filename == "test.txt"
        assert doc.created_at == datetime(2024, 1, 1)

    def test_create_with_defaults(self):
        doc = Document(id="doc-456", kb_id="kb-123", filename="test.txt")
        assert isinstance(doc.created_at, datetime)

    def test_missing_id_raises_error(self):
        with pytest.raises(ValidationError):
            Document(kb_id="kb-123", filename="test.txt")

    def test_missing_kb_id_raises_error(self):
        with pytest.raises(ValidationError):
            Document(id="doc-456", filename="test.txt")

    def test_missing_filename_raises_error(self):
        with pytest.raises(ValidationError):
            Document(id="doc-456", kb_id="kb-123")
