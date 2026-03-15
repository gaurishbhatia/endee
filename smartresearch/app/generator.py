"""
Generator — sends retrieved context + user question to Google Gemini to produce an answer.
"""
from typing import List, Dict, Any

import google.generativeai as genai

from app.config import GEMINI_API_KEY, GEMINI_MODEL

# Configure Gemini once at import time
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


SYSTEM_PROMPT = """\
You are SmartResearch, an AI research assistant.
Answer the user's question using ONLY the provided context chunks.
If the context does not contain enough information, say so honestly.
Always cite the source document and page number at the end of your answer.
Be concise, accurate, and helpful.
"""


def build_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a readable context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i} | Source: {chunk['source']} | Page: {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Generate a grounded answer using Gemini given the query and top-K retrieved chunks.

    Args:
        query:  The user's natural language question.
        chunks: Retrieved context chunks from Endee.

    Returns:
        The generated answer as a string.
    """
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Please copy .env.example to .env and add your key."
        )

    context = build_context(chunks)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== CONTEXT ===\n{context}\n\n"
        f"=== QUESTION ===\n{query}\n\n"
        f"=== ANSWER ==="
    )

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()
