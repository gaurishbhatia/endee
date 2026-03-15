"""
CLI interface for SmartResearch.

Usage:
    python -m cli.main ingest --source path/to/paper.pdf
    python -m cli.main ask "What is the main finding?"
    python -m cli.main ask "What is the main finding?" --top-k 3
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from app.rag_pipeline import RAGPipeline

console = Console()
pipeline = RAGPipeline()


@click.group()
def cli():
    """SmartResearch — AI Research Assistant powered by Endee Vector DB."""
    pass


@cli.command()
@click.option(
    "--source",
    "-s",
    required=True,
    help="Path to a PDF/TXT file or a web URL to ingest.",
)
def ingest(source: str):
    """Ingest a document into the Endee vector index."""
    try:
        count = pipeline.ingest(source)
        console.print(
            Panel(
                f"[bold green]✓ Successfully ingested[/bold green]\n"
                f"Source : [italic]{source}[/italic]\n"
                f"Chunks : {count}",
                title="[bold]SmartResearch[/bold]",
                border_style="green",
            )
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("question")
@click.option("--top-k", "-k", default=5, show_default=True, help="Number of chunks to retrieve.")
@click.option("--filter-source", "-f", default=None, help="Restrict search to a specific source filename.")
def ask(question: str, top_k: int, filter_source: str):
    """Ask a question against your ingested documents."""
    console.print(f"\n[bold cyan]Question:[/bold cyan] {question}\n")
    console.print("[dim]Retrieving from Endee and generating answer...[/dim]\n")

    try:
        answer, chunks = pipeline.ask(question, top_k=top_k, source_filter=filter_source)
    except EnvironmentError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    # Print answer
    console.print(
        Panel(
            Markdown(answer),
            title="[bold]SmartResearch Answer[/bold]",
            border_style="blue",
        )
    )

    # Print sources table
    if chunks:
        table = Table(title="Sources Retrieved from Endee", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Source", style="cyan")
        table.add_column("Page", justify="right")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Excerpt", style="white", max_width=60)

        for i, chunk in enumerate(chunks, 1):
            table.add_row(
                str(i),
                chunk["source"],
                str(chunk["page"]),
                f"{chunk['similarity']:.4f}",
                chunk["text"][:120] + "..." if len(chunk["text"]) > 120 else chunk["text"],
            )
        console.print(table)


if __name__ == "__main__":
    cli()
