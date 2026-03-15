"""
Unit tests for the retriever module.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch


class TestRetriever:
    @pytest.fixture(scope="class")
    def model(self):
        from sentence_transformers import SentenceTransformer
        from app.config import EMBEDDING_MODEL
        return SentenceTransformer(EMBEDDING_MODEL)

    def test_query_vector_shape(self, model):
        """The embedding model should produce a 384-dim vector for any query."""
        vec = model.encode(["What is RAG?"], normalize_embeddings=True)[0]
        assert vec.shape[0] == 384

    @patch("app.retriever.Endee")
    def test_retrieve_returns_normalised_dicts(self, MockEndee, model):
        """Retrieve should return clean dicts even if Endee returns objects."""
        from app.retriever import retrieve

        # Mock Endee result dicts
        mock_result = {
            "id": "abc123",
            "similarity": 0.89,
            "meta": {
                "text": "Endee is a vector database.",
                "source": "paper.pdf",
                "page": 2,
                "chunk_index": 0,
            }
        }

        mock_index = MagicMock()
        mock_index.query.return_value = [mock_result]

        mock_client = MagicMock()
        mock_client.get_index.return_value = mock_index
        MockEndee.return_value = mock_client

        results = retrieve("What is Endee?", top_k=1, model=model)

        assert len(results) == 1
        assert results[0]["id"] == "abc123"
        assert results[0]["similarity"] == 0.89
        assert results[0]["text"] == "Endee is a vector database."
        assert results[0]["source"] == "paper.pdf"
        assert results[0]["page"] == 2

    @patch("app.retriever.Endee")
    def test_retrieve_empty_results(self, MockEndee, model):
        """When Endee returns nothing, retrieve returns an empty list."""
        from app.retriever import retrieve

        mock_index = MagicMock()
        mock_index.query.return_value = []
        mock_client = MagicMock()
        mock_client.get_index.return_value = mock_index
        MockEndee.return_value = mock_client

        results = retrieve("completely unrelated query xyz", top_k=3, model=model)
        assert results == []
