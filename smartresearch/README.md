# 🔬 SmartResearch — AI Research Assistant powered by Endee

> **Retrieval-Augmented Generation (RAG)** — ask questions against your own documents using semantic search backed by the **Endee** vector database and answered by **Google Gemini**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Endee](https://img.shields.io/badge/Vector%20DB-Endee-6ee7f7?logo=database)](https://github.com/endee-io/endee)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%201.5%20Flash-a78bfa?logo=google)](https://ai.google.dev)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-f472b6?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

---

## 📌 Project Overview

**SmartResearch** lets you upload any PDF, text file, or web article and ask questions about it in plain English. Unlike traditional search, SmartResearch uses **semantic similarity** — it understands *meaning*, not just keywords.

### Problem Statement

Researchers, students, and analysts often need to extract insights from large document collections — academic papers, legal contracts, technical manuals. Reading everything in full is impractical. Keyword search misses context. LLMs hallucinate when operating from memory alone.

**SmartResearch solves this** by combining:
- **Document ingestion** — chunking and indexing your files
- **Semantic retrieval** — finding chunks that *mean* what you're asking
- **Grounded generation** — answering *only* from retrieved evidence, with sources cited

---

## 🏗️ System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         SmartResearch                           │
│                                                                 │
│  ┌──────────┐   chunk+embed   ┌──────────────────────────────┐  │
│  │ Document │ ──────────────► │     Endee Vector Database    │  │
│  │  Loader  │                 │  (cosine · INT8 · port 8080) │  │
│  └──────────┘                 └──────────────────────────────┘  │
│       ▲                                    │                    │
│       │                          top-K similarity search        │
│  PDF / TXT / URL                           │                    │
│                                            ▼                    │
│  ┌──────────────────┐    context    ┌────────────────┐          │
│  │  User Question   │ ────────────► │  Gemini 1.5    │          │
│  │  (UI or CLI)     │               │  Flash (LLM)   │          │
│  └──────────────────┘               └────────────────┘          │
│                                            │                    │
│                                    Grounded Answer +            │
│                                    Source Citations             │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1. Load | `document_loader.py` | Parse PDF/TXT/URL into page-level text |
| 2. Chunk | `ingest.py` | Split into 500-token overlapping chunks |
| 3. Embed | `ingest.py` | `all-MiniLM-L6-v2` → 384-dim float vectors |
| 4. Store | `ingest.py` | Upsert vectors + metadata into **Endee** |
| 5. Query embed | `retriever.py` | Embed user question with same model |
| 6. Search | `retriever.py` | Cosine similarity search in **Endee** → top-K chunks |
| 7. Generate | `generator.py` | Send chunks as context to **Gemini** → grounded answer |

---

## 🗄️ How Endee Is Used

Endee is the **core retrieval layer** of this project. Here's exactly how it's used:

### 1. Starting Endee (Docker)

```bash
docker compose up -d
# Endee dashboard → http://localhost:8080
```

### 2. Creating the Index

```python
from endee import Endee, Precision

client = Endee()  # connects to http://localhost:8080/api/v1
client.create_index(
    name="smartresearch_docs",
    dimension=384,          # all-MiniLM-L6-v2 output dim
    space_type="cosine",    # cosine similarity
    precision=Precision.INT8  # memory-efficient quantization
)
```

### 3. Storing Document Chunks

```python
index = client.get_index("smartresearch_docs")
index.upsert([
    {
        "id": "unique-chunk-id",
        "vector": [0.12, -0.05, ...],  # 384-dim embedding
        "meta": {
            "text": "Endee is a high-performance vector database...",
            "source": "paper.pdf",
            "page": 3,
            "chunk_index": 1,
        }
    }
])
```

### 4. Semantic Search at Query Time

```python
query_vec = model.encode(["What is RAG?"], normalize_embeddings=True)[0].tolist()
results = index.query(vector=query_vec, top_k=5)
# Each result: { id, similarity, meta: { text, source, page } }
```

**Endee features used:**
- ✅ Dense vector retrieval (cosine similarity)
- ✅ INT8 precision for memory efficiency
- ✅ Metadata (payload) storage and retrieval
- ✅ Docker deployment
- ✅ Official Python SDK (`pip install endee`)

---

## 📁 Project Structure

```
smartresearch/
├── .env                      # Your API keys (gitignored)
├── .env.example              # Template for environment variables
├── docker-compose.yml        # Spins up Endee server
├── requirements.txt          # Python dependencies
│
├── app/
│   ├── config.py             # Env var loading + constants
│   ├── document_loader.py    # PDF / TXT / URL loaders
│   ├── ingest.py             # Chunk → Embed → Upsert to Endee
│   ├── retriever.py          # Query embed → Endee search
│   ├── generator.py          # Gemini API integration
│   └── rag_pipeline.py       # Orchestrates the full RAG flow
│
├── cli/
│   └── main.py               # Click CLI (ingest + ask commands)
│
├── ui/
│   └── streamlit_app.py      # Streamlit web interface
│
├── data/
│   └── samples/
│       └── rag_overview.txt  # Sample document for demo
│
└── tests/
    ├── conftest.py
    ├── test_ingest.py        # Loader, chunking, embedding tests
    ├── test_retriever.py     # Query embed + Endee search tests
    └── test_pipeline.py      # End-to-end RAG pipeline tests
```

---

## ⚙️ Setup & Execution

### Prerequisites

- [Docker Desktop](https://docs.docker.com/get-docker/) installed and running
- Python 3.10+
- A free [Google Gemini API key](https://aistudio.google.com)

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/endee.git
cd endee/smartresearch
```

### Step 2 — Start Endee

**Option A: Docker Compose (recommended)**
```bash
docker compose up -d
```

**Option B: Direct Docker run**
```bash
docker run \
  --ulimit nofile=100000:100000 \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  --restart unless-stopped \
  endeeio/endee-server:latest
```

Verify Endee is running → open [http://localhost:8080](http://localhost:8080)

### Step 3 — Configure Environment

```bash
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
```

### Step 4 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 5 — Run SmartResearch

#### 🌐 Web UI (Streamlit)

```bash
streamlit run ui/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

1. Upload a PDF or TXT file in the sidebar (or enter a URL)
2. Click **⚡ Ingest Documents**
3. Type your question in the chat box

#### 💻 CLI

```bash
# Ingest a document
python -m cli.main ingest --source data/samples/rag_overview.txt

# Ask a question
python -m cli.main ask "What is Retrieval-Augmented Generation?"

# Restrict to a specific source file
python -m cli.main ask "What is Endee?" --filter-source rag_overview.txt

# Retrieve more chunks for complex questions
python -m cli.main ask "Explain the RAG pipeline step by step" --top-k 8
```

---

## 🧪 Running Tests

Tests are fully mocked — no running Endee or Gemini API needed.

```bash
cd smartresearch
pytest tests/ -v
```

Expected output:
```
tests/test_ingest.py::TestDocumentLoader::test_load_txt_file         PASSED
tests/test_ingest.py::TestDocumentLoader::test_load_document_dispatches_txt  PASSED
tests/test_ingest.py::TestDocumentLoader::test_load_document_missing_file    PASSED
tests/test_ingest.py::TestDocumentLoader::test_load_document_unsupported_extension  PASSED
tests/test_ingest.py::TestChunking::test_chunk_pages_produces_multiple_chunks  PASSED
tests/test_ingest.py::TestChunking::test_chunk_metadata_preserved    PASSED
tests/test_ingest.py::TestChunking::test_chunk_text_non_empty        PASSED
tests/test_ingest.py::TestEmbedding::test_embed_chunks_adds_vector   PASSED
tests/test_ingest.py::TestEmbedding::test_embed_chunks_batch         PASSED
tests/test_ingest.py::TestEndeeIntegration::test_ensure_index_creates_when_missing  PASSED
tests/test_ingest.py::TestEndeeIntegration::test_ensure_index_skips_when_existing   PASSED
tests/test_retriever.py::TestRetriever::test_query_vector_shape      PASSED
tests/test_retriever.py::TestRetriever::test_retrieve_returns_normalised_dicts  PASSED
tests/test_retriever.py::TestRetriever::test_retrieve_empty_results  PASSED
tests/test_pipeline.py::TestRAGPipeline::test_ingest_returns_chunk_count  PASSED
tests/test_pipeline.py::TestRAGPipeline::test_ask_returns_answer_and_sources  PASSED
tests/test_pipeline.py::TestRAGPipeline::test_ask_no_documents_returns_fallback  PASSED
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vector Database** | [Endee](https://github.com/endee-io/endee) | Store & retrieve 384-dim document embeddings |
| **Embeddings** | `sentence-transformers` (MiniLM-L6-v2) | Local, free, 384-dim semantic vectors |
| **LLM** | Google Gemini 1.5 Flash | Grounded answer generation |
| **Document Parsing** | PyPDF2, BeautifulSoup4 | PDF and web page extraction |
| **Chunking** | LangChain Text Splitter | Recursive 500-token chunks with overlap |
| **Web UI** | Streamlit | Chat interface with source attribution |
| **CLI** | Click + Rich | Terminal interface with formatted output |
| **Tests** | pytest + unittest.mock | Fully mocked unit + integration tests |

---

## 🔒 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key (required) | — |
| `ENDEE_BASE_URL` | Endee server URL | `http://localhost:8080/api/v1` |
| `ENDEE_AUTH_TOKEN` | Endee auth token (optional) | empty |
| `ENDEE_INDEX_NAME` | Name of the Endee index | `smartresearch_docs` |
| `EMBEDDING_MODEL` | Sentence-Transformers model name | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Tokens per chunk | `500` |
| `CHUNK_OVERLAP` | Overlap tokens between chunks | `50` |
| `TOP_K_RESULTS` | Chunks retrieved per query | `5` |

---

## 📜 License

This project is built on top of the [Endee](https://github.com/endee-io/endee) open-source vector database, licensed under the **Apache License 2.0**.

---

## 🙏 Acknowledgements

- [Endee](https://endee.io) — open-source vector database
- [Sentence-Transformers](https://www.sbert.net) — local embedding models
- [Google Gemini](https://ai.google.dev) — LLM generation API
- Lewis et al., *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"*, NeurIPS 2020
