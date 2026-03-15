"""
End-to-end pipeline tests (Endee and Gemini are mocked).
"""
import sys
from pathlib import Path
import tempfile, os

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch


SAMPLE_DOC = (
    "Endee is an open-source, high-performance vector database designed for AI applications. "
    "It supports dense vector retrieval, sparse search, and metadata-aware payload filtering. "
    "Endee ships with flexible deployment paths including Docker and manual builds. "
    "RAG (Retrieval-Augmented Generation) uses a vector database to retrieve relevant document "
    "chunks that are fed as context to a large language model to produce grounded answers. "
    "This project, SmartResearch, implements a full RAG pipeline using Endee and Google Gemini."
) * 4


def make_txt(content=SAMPLE_DOC):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name


class TestRAGPipeline:
    @patch("app.ingest.Endee")
    def test_ingest_returns_chunk_count(self, MockEndee):
        """Ingest should return a positive integer chunk count."""
        from app.rag_pipeline import RAGPipeline

        mock_index = MagicMock()
        mock_client = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.get_index.return_value = mock_index
        MockEndee.return_value = mock_client

        pipeline = RAGPipeline()
        path = make_txt()
        try:
            count = pipeline.ingest(path)
            assert count > 0, "Should return positive chunk count"
        finally:
            os.unlink(path)

    @patch("app.generator.genai")
    @patch("app.retriever.Endee")
    def test_ask_returns_answer_and_sources(self, MockEndee, MockGenai):
        """Ask should return an answer string and a list of source chunks."""
        from app.rag_pipeline import RAGPipeline

        # Mock Endee retrieval
        mock_result = {
            "id": "chunk-1",
            "similarity": 0.92,
            "meta": {
                "text": "Endee is used as a vector database in this RAG pipeline.",
                "source": "test.txt",
                "page": 1,
                "chunk_index": 0,
            }
        }
        mock_index = MagicMock()
        mock_index.query.return_value = [mock_result]
        mock_client = MagicMock()
        mock_client.get_index.return_value = mock_index
        MockEndee.return_value = mock_client

        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.text = "Endee is an open-source vector database used for semantic retrieval."
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = mock_response
        MockGenai.GenerativeModel.return_value = mock_model_instance

        pipeline = RAGPipeline()
        answer, sources = pipeline.ask("What is Endee?", top_k=1)

        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(sources, list)
        assert len(sources) > 0
        assert "text" in sources[0]
        assert "source" in sources[0]

    @patch("app.retriever.Endee")
    def test_ask_no_documents_returns_fallback(self, MockEndee):
        """When no chunks found, ask should return a helpful fallback message."""
        from app.rag_pipeline import RAGPipeline

        mock_index = MagicMock()
        mock_index.query.return_value = []
        mock_client = MagicMock()
        mock_client.get_index.return_value = mock_index
        MockEndee.return_value = mock_client

        pipeline = RAGPipeline()
        answer, sources = pipeline.ask("What is RAG?")

        assert "ingest" in answer.lower() or "no relevant" in answer.lower()
        assert sources == []
