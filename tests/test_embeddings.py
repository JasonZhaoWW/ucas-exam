from unittest.mock import MagicMock, patch

import numpy as np

from src.api.embeddings import embed_query, embed_texts


class TestEmbedTexts:
    def test_empty_input_returns_empty_list(self):
        assert embed_texts([]) == []

    @patch("src.api.embeddings._get_model")
    def test_single_text_returns_one_vector(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_get_model.return_value = mock_model

        result = embed_texts(["hello world"])

        assert len(result) == 1
        assert len(result[0]) == 3
        mock_model.encode.assert_called_once_with(["hello world"])

    @patch("src.api.embeddings._get_model")
    def test_multiple_texts_returns_matching_count(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )
        mock_get_model.return_value = mock_model

        result = embed_texts(["alpha", "beta", "gamma"])

        assert len(result) == 3
        assert all(len(v) == 3 for v in result)


class TestEmbedQuery:
    @patch("src.api.embeddings._get_model")
    def test_returns_single_flat_vector(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_get_model.return_value = mock_model

        result = embed_query("search term")

        assert isinstance(result, list)
        assert len(result) == 3
        mock_model.encode.assert_called_once_with(["search term"])


class TestLazyLoading:
    def setup_method(self):
        import src.api.embeddings as mod

        mod._model = None

    @patch("src.api.embeddings.SentenceTransformer")
    def test_model_not_loaded_at_import(self, mock_cls):
        import src.api.embeddings as mod

        assert mod._model is None
        mock_cls.assert_not_called()

    @patch("src.api.embeddings.SentenceTransformer")
    def test_model_loaded_on_first_call(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_cls.return_value = mock_instance

        embed_texts(["hello"])

        mock_cls.assert_called_once()

    @patch("src.api.embeddings.SentenceTransformer")
    def test_model_reused_on_subsequent_calls(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_cls.return_value = mock_instance

        embed_texts(["hello"])
        embed_texts(["world"])

        mock_cls.assert_called_once()


class TestModelNameConfig:
    def setup_method(self):
        import src.api.embeddings as mod

        mod._model = None

    @patch("src.api.embeddings.SentenceTransformer")
    def test_default_model_name(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.array([[0.1]])
        mock_cls.return_value = mock_instance

        embed_texts(["x"])

        mock_cls.assert_called_once_with("BAAI/bge-m3")

    @patch("src.api.embeddings.SentenceTransformer")
    def test_model_name_from_env(self, mock_cls, monkeypatch):
        monkeypatch.setenv("KB_EMBEDDING_MODEL", "custom/model")
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.array([[0.1]])
        mock_cls.return_value = mock_instance

        embed_texts(["x"])

        mock_cls.assert_called_once_with("custom/model")
