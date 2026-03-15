"""
Streamlit web UI for SmartResearch.
Run with: streamlit run ui/streamlit_app.py
"""
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartResearch — AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* Header */
.hero-header {
    text-align: center;
    padding: 2rem 0 1rem;
}
.hero-header h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6ee7f7, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-header p {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

/* Cards */
.answer-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(110,231,247,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}

/* Source chips */
.source-chip {
    display: inline-block;
    background: rgba(167,139,250,0.15);
    border: 1px solid rgba(167,139,250,0.4);
    color: #a78bfa;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    margin: 2px;
}

/* Metric boxes */
.metric-box {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.07);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15,15,26,0.95) !important;
    border-right: 1px solid rgba(110,231,247,0.1);
}

/* Input */
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(110,231,247,0.25) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(135deg, #6ee7f7, #a78bfa) !important;
    color: #0f0f1a !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(110,231,247,0.35) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Lazy load pipeline ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading SmartResearch engine...")
def load_pipeline():
    from app.rag_pipeline import RAGPipeline
    return RAGPipeline()


# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_sources" not in st.session_state:
    st.session_state.ingested_sources = []


# ── Sidebar — Document Ingestion ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Document Ingestion")
    st.markdown("Upload documents to build your knowledge base.")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    url_input = st.text_input("Or enter a web URL", placeholder="https://example.com/article")

    ingest_btn = st.button("⚡ Ingest Documents", use_container_width=True)

    if ingest_btn:
        pipeline = load_pipeline()
        sources_to_ingest = []

        # Save uploaded files to temp and collect paths
        for uf in (uploaded_files or []):
            suffix = Path(uf.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.read())
                sources_to_ingest.append((tmp.name, uf.name))

        # Add URL
        if url_input.strip():
            sources_to_ingest.append((url_input.strip(), url_input.strip()))

        if not sources_to_ingest:
            st.warning("Please upload a file or enter a URL first.")
        else:
            for src_path, display_name in sources_to_ingest:
                with st.spinner(f"Ingesting {display_name}..."):
                    try:
                        count = pipeline.ingest(src_path)
                        st.success(f"✓ {display_name} → {count} chunks indexed")
                        if display_name not in st.session_state.ingested_sources:
                            st.session_state.ingested_sources.append(display_name)
                    except Exception as e:
                        st.error(f"Failed: {e}")

    st.divider()
    st.markdown("### 📚 Indexed Sources")
    if st.session_state.ingested_sources:
        for src in st.session_state.ingested_sources:
            st.markdown(f'<span class="source-chip">📄 {Path(src).name if "/" not in src and "\\\\" not in src else src}</span>', unsafe_allow_html=True)
    else:
        st.caption("No documents ingested yet.")

    st.divider()
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, 5)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🔬 SmartResearch</h1>
    <p>AI-powered research assistant · Semantic search backed by <strong>Endee</strong> vector database</p>
</div>
""", unsafe_allow_html=True)

# Metrics row
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-box">
        <div style="color:#6ee7f7;font-size:1.6rem;font-weight:700">{len(st.session_state.ingested_sources)}</div>
        <div style="color:#64748b;font-size:0.85rem">Sources Indexed</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-box">
        <div style="color:#a78bfa;font-size:1.6rem;font-weight:700">{len(st.session_state.messages)}</div>
        <div style="color:#64748b;font-size:0.85rem">Questions Asked</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-box">
        <div style="color:#f472b6;font-size:1.6rem;font-weight:700">Endee</div>
        <div style="color:#64748b;font-size:0.85rem">Vector Database</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🔬"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources retrieved from Endee"):
                for i, chunk in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**Chunk {i}** · `{chunk['source']}` · Page {chunk['page']} · "
                        f"Score: `{chunk['similarity']:.4f}`\n\n> {chunk['text'][:300]}..."
                    )

# Question input
question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🔬"):
        with st.spinner("Searching Endee and generating answer..."):
            pipeline = load_pipeline()
            try:
                answer, chunks = pipeline.ask(question, top_k=top_k)
            except EnvironmentError as e:
                answer = f"⚠️ **Configuration Error:** {e}"
                chunks = []
            except Exception as e:
                answer = f"⚠️ **Error:** {e}"
                chunks = []

        st.markdown(answer)

        if chunks:
            with st.expander("📎 Sources retrieved from Endee"):
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(
                        f"**Chunk {i}** · `{chunk['source']}` · Page {chunk['page']} · "
                        f"Score: `{chunk['similarity']:.4f}`\n\n> {chunk['text'][:300]}..."
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks,
    })
