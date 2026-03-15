"""
Unit tests for the ingest pipeline.
Run with: pytest tests/ -v
"""
import sys
from pathlib import Path
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_txt_file(content: str) -> str:
    """Write content to a temp .txt file and return the path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name


SAMPLE_TEXT = """
Artificial intelligence (AI) is transforming every industry. Machine learning models
are capable of understanding human language through natural language processing (NLP).
Vector databases, such as Endee, store high-dimensional embeddings that enable semantic
similarity search at scale. This allows AI systems to retrieve the most relevant context
before generating responses, a technique known as Retrieval-Augmented Generation (RAG).
""" * 5  # repeat to ensure multiple chunks


# ── Tests: document_loader ─────────────────────────────────────────────────────

class TestDocumentLoader:
    def test_load_txt_file(self):
        from app.document_loader import load_txt
        path = make_txt_file("Hello world. This is a test document.")
        try:
            pages = load_txt(path)
            assert len(pages) == 1
            assert "Hello world" in pages[0]["text"]
            assert pages[0]["page"] == 1
        finally:
            os.unlink(path)

    def test_load_document_dispatches_txt(self):
        from app.document_loader import load_document
        path = make_txt_file("Test content for SmartResearch.")
        try:
            pages = load_document(path)
            assert isinstance(pages, list)
            assert len(pages) > 0
        finally:
            os.unlink(path)

    def test_load_document_missing_file(self):
        from app.document_loader import load_document
        with pytest.raises(FileNotFoundError):
            load_document("/nonexistent/file.txt")

    def test_load_document_unsupported_extension(self):
        from app.document_loader import load_document
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xyz")
        tmp.close()
        try:
            with pytest.raises(ValueError):
                load_document(tmp.name)
        finally:
            os.unlink(tmp.name)


# ── Tests: chunking ───────────────────────────────────────────────────────────

class TestChunking:
    def test_chunk_pages_produces_multiple_chunks(self):
        from app.ingest import chunk_pages
        pages = [{"text": SAMPLE_TEXT, "source": "test.txt", "page": 1}]
        chunks = chunk_pages(pages)
        assert len(chunks) > 1, "Long text should produce multiple chunks"

    def test_chunk_metadata_preserved(self):
        from app.ingest import chunk_pages
        pages = [{"text": SAMPLE_TEXT, "source": "paper.pdf", "page": 3}]
        chunks = chunk_pages(pages)
        for chunk in chunks:
            assert chunk["source"] == "paper.pdf"
            assert chunk["page"] == 3
            assert "text" in chunk
            assert "chunk_index" in chunk

    def test_chunk_text_non_empty(self):
        from app.ingest import chunk_pages
        pages = [{"text": SAMPLE_TEXT, "source": "doc.txt", "page": 1}]
        chunks = chunk_pages(pages)
        for chunk in chunks:
            assert chunk["text"].strip() != ""


# ── Tests: embedding ──────────────────────────────────────────────────────────

class TestEmbedding:
    @pytest.fixture(scope="class")
    def model(self):
        from sentence_transformers import SentenceTransformer
        from app.config import EMBEDDING_MODEL
        return SentenceTransformer(EMBEDDING_MODEL)

    def test_embed_chunks_adds_vector(self, model):
        from app.ingest import embed_chunks
        chunks = [{"text": "Endee is a vector database.", "source": "t.txt", "page": 1, "chunk_index": 0}]
        result = embed_chunks(chunks, model)
        assert "vector" in result[0]
        assert len(result[0]["vector"]) == 384  # all-MiniLM-L6-v2 dim

    def test_embed_chunks_batch(self, model):
        from app.ingest import embed_chunks
        chunks = [
            {"text": f"Text chunk number {i}", "source": "t.txt", "page": 1, "chunk_index": i}
            for i in range(5)
        ]
        result = embed_chunks(chunks, model)
        assert len(result) == 5
        assert all("vector" in c for c in result)


# ── Tests: Endee client (mocked) ──────────────────────────────────────────────

class TestEndeeIntegration:
    @patch("app.ingest.Endee")
    def test_ensure_index_creates_when_missing(self, MockEndee):
        from app.ingest import ensure_index
        mock_client = MagicMock()
        mock_client.list_indexes.return_value = []  # no existing indexes
        mock_index = MagicMock()
        mock_client.get_index.return_value = mock_index

        MockEndee.return_value = mock_client

        result = ensure_index(mock_client)

        mock_client.create_index.assert_called_once()
        mock_client.get_index.assert_called_once()

    @patch("app.ingest.Endee")
    def test_ensure_index_skips_when_existing(self, MockEndee):
        from app.config import ENDEE_INDEX_NAME
        from app.ingest import ensure_index
        mock_client = MagicMock()
        existing = MagicMock()
        existing.name = ENDEE_INDEX_NAME
        mock_client.list_indexes.return_value = [existing]
        mock_client.get_index.return_value = MagicMock()

        ensure_index(mock_client)

        mock_client.create_index.assert_not_called()
