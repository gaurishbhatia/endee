"""
Ingest pipeline — loads documents, splits into chunks, embeds, and upserts to Endee.
"""
import uuid
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from endee import Endee, Precision
from tqdm import tqdm
from rich.console import Console

from app.config import (
    ENDEE_BASE_URL,
    ENDEE_AUTH_TOKEN,
    ENDEE_INDEX_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from app.document_loader import load_document

console = Console()


def get_endee_client() -> Endee:
    """Create and configure an Endee client."""
    token = ENDEE_AUTH_TOKEN if ENDEE_AUTH_TOKEN else None
    client = Endee(token) if token else Endee()
    client.set_base_url(ENDEE_BASE_URL)
    return client


def ensure_index(client: Endee) -> Any:
    """Create the Endee index if it doesn't already exist, then return it."""
    indexes = client.list_indexes()
    # Endee list_indexes returns a list of strings (index names)
    existing = [idx if isinstance(idx, str) else getattr(idx, "name", str(idx)) for idx in indexes]
    
    if ENDEE_INDEX_NAME not in existing:
        console.print(f"[cyan]Creating Endee index:[/cyan] [bold]{ENDEE_INDEX_NAME}[/bold]")
        client.create_index(
            name=ENDEE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            space_type="cosine",
            precision=Precision.INT8,
        )
    else:
        console.print(f"[green]Using existing index:[/green] [bold]{ENDEE_INDEX_NAME}[/bold]")
    return client.get_index(name=ENDEE_INDEX_NAME)


def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split page-level text into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "source": page["source"],
                "page": page["page"],
                "chunk_index": i,
            })
    return chunks


def embed_chunks(
    chunks: List[Dict[str, Any]],
    model: SentenceTransformer,
) -> List[Dict[str, Any]]:
    """Add embedding vectors to each chunk dict."""
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    for chunk, vec in zip(chunks, vectors):
        chunk["vector"] = vec.tolist()
    return chunks


def ingest(source: str, model: Optional[SentenceTransformer] = None) -> int:
    """
    Full ingest pipeline for a single source (file path or URL).
    Returns the number of chunks stored.
    """
    console.print(f"\n[bold blue]SmartResearch Ingest[/bold blue]")
    console.print(f"Source: [italic]{source}[/italic]")

    # 1. Load
    console.print("[cyan]Loading document...[/cyan]")
    pages = load_document(source)
    console.print(f"  → {len(pages)} page(s) loaded")

    # 2. Chunk
    console.print("[cyan]Splitting into chunks...[/cyan]")
    chunks = chunk_pages(pages)
    console.print(f"  → {len(chunks)} chunk(s) created")

    # 3. Embed
    if model is None:
        console.print(f"[cyan]Loading embedding model ({EMBEDDING_MODEL})...[/cyan]")
        model = SentenceTransformer(EMBEDDING_MODEL)
    console.print("[cyan]Embedding chunks...[/cyan]")
    chunks = embed_chunks(chunks, model)

    # 4. Upsert to Endee
    console.print("[cyan]Connecting to Endee...[/cyan]")
    client = get_endee_client()
    index = ensure_index(client)

    batch_size = 64
    total_upserted = 0
    console.print(f"[cyan]Upserting {len(chunks)} vectors to Endee...[/cyan]")
    for i in tqdm(range(0, len(chunks), batch_size), desc="Upserting"):
        batch = chunks[i : i + batch_size]
        vectors = [
            {
                "id": str(uuid.uuid4()),
                "vector": c["vector"],
                "meta": {
                    "text": c["text"],
                    "source": c["source"],
                    "page": c["page"],
                    "chunk_index": c["chunk_index"],
                },
            }
            for c in batch
        ]
        index.upsert(vectors)
        total_upserted += len(batch)

    console.print(f"[bold green]✓ Done![/bold green] {total_upserted} chunks indexed from '{source}'")
    return total_upserted
