"""
RAG pipeline — orchestrates document ingest and question answering.
"""
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL, TOP_K_RESULTS
from app.ingest import ingest
from app.retriever import retrieve
from app.generator import generate_answer


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline backed by Endee.

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest("path/to/paper.pdf")
        answer, sources = pipeline.ask("What is the main contribution of this paper?")
    """

    def __init__(self):
        # Load the embedding model once and reuse across calls
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    # ── Ingest ────────────────────────────────────────────────────────────────

    def ingest(self, source: str) -> int:
        """
        Ingest a document source into Endee.

        Args:
            source: File path (.pdf / .txt) or a web URL.

        Returns:
            Number of chunks stored.
        """
        return ingest(source, model=self.model)

    # ── Ask ───────────────────────────────────────────────────────────────────

    def ask(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        source_filter: Optional[str] = None,
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Answer a natural language question using RAG.

        Args:
            query:         The user's question.
            top_k:         Number of chunks to retrieve from Endee.
            source_filter: Restrict retrieval to a specific source filename.

        Returns:
            (answer: str, sources: list of chunk dicts)
        """
        # 1. Retrieve relevant chunks from Endee
        chunks = retrieve(query, top_k=top_k, model=self.model, source_filter=source_filter)

        if not chunks:
            return (
                "No relevant documents found. Please ingest some documents first.",
                [],
            )

        # 2. Generate answer with Gemini
        answer = generate_answer(query, chunks)
        return answer, chunks
