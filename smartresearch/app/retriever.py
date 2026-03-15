"""
Retriever — embeds a user query and searches Endee for the most relevant chunks.

Endee SDK (v0.1.18) API notes:
  - filter must be a LIST of dicts: [{"field": {"$eq": value}}]
  - query() returns a list of dicts, NOT objects
"""
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
from endee import Endee

from app.config import (
    ENDEE_BASE_URL,
    ENDEE_AUTH_TOKEN,
    ENDEE_INDEX_NAME,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
)


def get_endee_client() -> Endee:
    token = ENDEE_AUTH_TOKEN if ENDEE_AUTH_TOKEN else None
    client = Endee(token) if token else Endee()
    client.set_base_url(ENDEE_BASE_URL)
    return client


def retrieve(
    query: str,
    top_k: int = TOP_K_RESULTS,
    model: Optional[SentenceTransformer] = None,
    source_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Embed a query and return the top-K most similar chunks from Endee.

    Args:
        query:         Natural language question or search phrase.
        top_k:         Number of results to return.
        model:         Pre-loaded SentenceTransformer (loaded on demand if None).
        source_filter: Optional source filename to restrict search scope.

    Returns:
        List of dicts with keys: id, similarity, text, source, page, chunk_index
    """
    # Embed the query
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL)
    query_vector = model.encode([query], normalize_embeddings=True)[0].tolist()

    # Endee filter MUST be a list of dicts — e.g. [{"source": {"$eq": "file.pdf"}}]
    filters = None
    if source_filter:
        filters = [{"source": {"$eq": source_filter}}]

    # Query Endee
    client = get_endee_client()
    index = client.get_index(name=ENDEE_INDEX_NAME)

    # Endee query() returns a list of dicts (not objects)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        filter=filters,
    )

    chunks = []
    for r in results:
        meta = r.get("meta") or {}
        chunks.append({
            "id": r.get("id", ""),
            "similarity": round(float(r.get("similarity", 0.0)), 4),
            "text": meta.get("text", ""),
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", 0),
            "chunk_index": meta.get("chunk_index", 0),
        })

    return chunks
