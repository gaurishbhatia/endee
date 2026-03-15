"""
Document loaders — supports PDF, plain text, and web URLs.
Returns a list of dicts: {"text": str, "source": str, "page": int}
"""
from pathlib import Path
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup


def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extract text page-by-page from a PDF."""
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("Install PyPDF2: pip install PyPDF2")

    pages = []
    path = Path(file_path)
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({
                    "text": text,
                    "source": path.name,
                    "page": i + 1,
                })
    return pages


def load_txt(file_path: str) -> List[Dict[str, Any]]:
    """Load a plain text file as a single document."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    return [{"text": text, "source": path.name, "page": 1}]


def load_url(url: str) -> List[Dict[str, Any]]:
    """Fetch a web page and extract its visible text."""
    headers = {"User-Agent": "SmartResearch/1.0"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Remove script / style tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n").strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    text = "\n".join(lines)
    return [{"text": text, "source": url, "page": 1}]


def load_document(source: str) -> List[Dict[str, Any]]:
    """
    Auto-detect source type and load accordingly.
    source: local file path (PDF / TXT) or http(s) URL
    """
    if source.startswith("http://") or source.startswith("https://"):
        return load_url(source)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {source}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(source)
    elif suffix in (".txt", ".md", ".rst"):
        return load_txt(source)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .txt, .md")
