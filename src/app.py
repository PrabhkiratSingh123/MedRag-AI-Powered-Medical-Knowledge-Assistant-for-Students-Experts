"""
MedRAG — Medical AI Assistant
------------------------------
LLM + RAG pipeline for medical students and healthcare professionals.
Powered by Groq LLM · MongoDB Atlas Vector Search · HuggingFace Embeddings.

Run with:
    streamlit run src/app.py
"""

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.config import (
    DOCS_DIR, GROQ_MODEL, TOP_K,
    MONGODB_DB_NAME, MONGODB_COLLECTION_NAME,
    validate_config, get_mongo_client,
)
from src.guardrails import check_query
from src.retriever import retrieve_and_rerank, retrieve
from src.generator import generate_stream
from src.guardrails import check_grounding, inject_disclaimer

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedRAG — Medical AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "MedRAG: Evidence-based Medical AI for students and professionals. Powered by Groq + MongoDB Atlas + RAG.",
    },
)

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #060d18 0%, #0a1628 55%, #080f1e 100%);
    min-height: 100vh;
}
.main .block-container { padding: 2rem 2rem 5rem 2rem; max-width: 1080px; }

/* ── Hero Header ── */
.hero {
    text-align: center;
    padding: 2.8rem 1.5rem 2rem;
    background: linear-gradient(135deg, rgba(0,180,216,0.07), rgba(100,60,220,0.07));
    border: 1px solid rgba(0,180,216,0.14);
    border-radius: 22px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(0,180,216,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero h1 {
    font-size: 3rem; font-weight: 700; margin: 0 0 0.3rem;
    background: linear-gradient(90deg, #00c8f0, #8b5cf6, #00c8f0);
    background-size: 200%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 5s linear infinite;
}
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }
.hero .tagline { color: #7a93b0; font-size: 1.05rem; margin: 0 0 1.2rem; }
.badge {
    display: inline-block;
    background: rgba(0,180,216,0.1); border: 1px solid rgba(0,180,216,0.25);
    color: #00b4d8; border-radius: 20px; padding: 3px 13px;
    font-size: 0.73rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; margin: 0 4px;
}
.badge.purple {
    background: rgba(139,92,246,0.1); border-color: rgba(139,92,246,0.25); color: #a78bfa;
}
.badge.green {
    background: rgba(16,185,129,0.1); border-color: rgba(16,185,129,0.25); color: #34d399;
}

/* ── Chat messages ── */
.chat-msg {
    display: flex; gap: 14px; padding: 1.1rem 1.3rem;
    border-radius: 18px; margin-bottom: 1rem;
    animation: fadeUp 0.3s ease;
}
@keyframes fadeUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
.chat-msg.user {
    background: linear-gradient(135deg, rgba(0,180,216,0.09), rgba(0,180,216,0.04));
    border: 1px solid rgba(0,180,216,0.18);
    flex-direction: row-reverse;
}
.chat-msg.assistant {
    background: linear-gradient(135deg, rgba(139,92,246,0.09), rgba(139,92,246,0.03));
    border: 1px solid rgba(139,92,246,0.18);
}
.chat-msg.emergency {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(239,68,68,0.04));
    border: 1px solid rgba(239,68,68,0.35);
}
.chat-msg.blocked {
    background: linear-gradient(135deg, rgba(245,158,11,0.09), rgba(245,158,11,0.03));
    border: 1px solid rgba(245,158,11,0.25);
}
.avatar { font-size: 1.5rem; flex-shrink: 0; width: 38px; text-align: center; padding-top: 2px; }
.msg-body { flex: 1; color: #cdd8e8; line-height: 1.7; font-size: 0.96rem; }
.msg-body.user-text { text-align: right; color: #9ab8d4; }

/* ── Source cards ── */
.src-card {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 0.7rem 1rem; margin-bottom: 0.5rem;
    font-size: 0.84rem; color: #8b9ab0; transition: border-color 0.2s, background 0.2s;
}
.src-card:hover { border-color: rgba(0,180,216,0.35); background: rgba(0,180,216,0.04); }
.src-title { color: #00b4d8; font-weight: 600; font-size: 0.86rem; }
.src-score {
    float: right; background: rgba(0,180,216,0.12); color: #00b4d8;
    padding: 1px 9px; border-radius: 10px; font-size: 0.74rem;
    font-family: 'JetBrains Mono', monospace;
}
.src-preview { margin-top: 0.35rem; font-style: italic; line-height: 1.5; }

/* ── Grounding warning ── */
.ground-warn {
    background: rgba(245,158,11,0.07); border: 1px solid rgba(245,158,11,0.25);
    border-radius: 8px; padding: 0.45rem 0.8rem; color: #fbbf24;
    font-size: 0.81rem; margin-top: 0.5rem;
}

/* ── Info pills (sidebar stats) ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 0.7rem 1rem;
}
[data-testid="stMetricLabel"] { color: #7a93b0 !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: #00b4d8 !important; font-size: 1.4rem !important; font-weight: 700 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1322 0%, #060d18 100%);
    border-right: 1px solid rgba(255,255,255,0.055);
}
[data-testid="stSidebar"] .stMarkdown { color: #7a93b0; }

/* ── Input ── */
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.045) !important;
    border: 1px solid rgba(0,180,216,0.22) !important;
    border-radius: 12px !important; color: #cdd8e8 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: rgba(0,180,216,0.55) !important;
    box-shadow: 0 0 0 3px rgba(0,180,216,0.1) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00b4d8, #7c3aed);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; padding: 0.5rem 1.4rem;
    transition: opacity 0.2s, transform 0.15s;
}
.stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }

/* ── Toggle / slider ── */
.stSlider [data-testid="stThumbValue"] { color: #00b4d8 !important; }

hr { border-color: rgba(255,255,255,0.06) !important; }
.stSpinner > div { border-top-color: #00b4d8 !important; }

/* ── Suggestions bar ── */
.suggestions { color: #4a6070; font-size: 0.82rem; margin-top: 0.4rem; }
.suggestions em { color: #5a7a90; }

/* ── Quick-action chips ── */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1rem; }
.chip {
    background: rgba(0,180,216,0.07); border: 1px solid rgba(0,180,216,0.2);
    color: #00b4d8; border-radius: 20px; padding: 4px 14px;
    font-size: 0.8rem; cursor: pointer; transition: background 0.2s;
    display: inline-block;
}
.chip:hover { background: rgba(0,180,216,0.15); }
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages": [],
        "total_queries": 0,
        "sources_used": set(),
        "uploaded_filenames": set(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ RAG Settings")

    top_k = st.slider(
        "Context chunks to retrieve", 1, 10, TOP_K, 1,
        help="More chunks = richer context but slightly slower."
    )
    show_sources   = st.toggle("Show source excerpts",           value=True)
    show_grounding = st.toggle("Show grounding warnings",        value=True)
    use_reranker   = st.toggle("Enable cross-encoder reranking", value=True)

    st.markdown("---")
    st.markdown("## 📥 Upload Medical Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, CSV, JSON",
        type=["pdf", "txt", "csv", "json", "jsonl"],
        accept_multiple_files=True,
        help="Your documents are embedded and stored in MongoDB Atlas for retrieval.",
    )
    if uploaded_files:
        if st.button("🚀 Ingest to MongoDB Atlas", use_container_width=True):
            with st.spinner("Embedding & saving to MongoDB Atlas…"):
                docs_dir = Path(DOCS_DIR)
                docs_dir.mkdir(parents=True, exist_ok=True)
                for u_file in uploaded_files:
                    fp = docs_dir / u_file.name
                    with open(fp, "wb") as f:
                        f.write(u_file.getbuffer())
                    st.session_state.uploaded_filenames.add(u_file.name)
                from src.ingestion import run_ingestion
                run_ingestion(docs_dir=str(docs_dir))
            st.success(f"✅ Ingested {len(uploaded_files)} file(s) into MongoDB Atlas!")

    if st.session_state.uploaded_filenames:
        st.markdown(
            f"<small style='color:#00b4d8;'>Indexed: {', '.join(st.session_state.uploaded_filenames)}</small>",
            unsafe_allow_html=True,
        )
        if st.button("🧹 Remove Session Docs from DB", use_container_width=True):
            from src.ingestion import delete_uploaded_documents
            delete_uploaded_documents(list(st.session_state.uploaded_filenames))
            st.session_state.uploaded_filenames = set()
            st.success("🧹 Removed session documents.")
            st.rerun()

    st.markdown("---")
    st.markdown("## 📊 Session Stats")
    c1, c2 = st.columns(2)
    with c1: st.metric("Queries", st.session_state.total_queries)
    with c2: st.metric("Sources", len(st.session_state.sources_used))

    st.markdown("---")
    st.markdown("## 🤖 Pipeline Info")
    st.markdown(f"""
| | |
|---|---|
| **LLM** | `{GROQ_MODEL}` |
| **Embeddings** | MiniLM-L6-v2 |
| **Vector DB** | MongoDB Atlas |
| **Reranker** | ms-marco-MiniLM |
""")

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        if st.session_state.uploaded_filenames:
            from src.ingestion import delete_uploaded_documents
            delete_uploaded_documents(list(st.session_state.uploaded_filenames))
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.sources_used = set()
        st.session_state.uploaded_filenames = set()
        st.rerun()

    cfg_warnings = validate_config()
    if cfg_warnings:
        st.markdown("---")
        for w in cfg_warnings:
            st.warning(w)

    st.markdown("---")
    st.markdown(
        "<small style='color:#3a5060;'>MedRAG v3.0 · Educational use only</small>",
        unsafe_allow_html=True,
    )


# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🩺 MedRAG</h1>
    <p class="tagline">AI-Powered Medical Knowledge Assistant for Students &amp; Experts</p>
    <span class="badge">Groq LLM</span>
    <span class="badge purple">RAG Pipeline</span>
    <span class="badge green">MongoDB Atlas</span>
    <span class="badge">HuggingFace</span>
</div>
""", unsafe_allow_html=True)


# ─── Message Renderer ─────────────────────────────────────────────────────────
def render_message(role: str, content: str, sources=None,
                   grounding=None, is_emergency=False, is_blocked=False):
    if is_emergency:
        css, avatar = "emergency", "🚨"
    elif is_blocked:
        css, avatar = "blocked", "⚠️"
    elif role == "user":
        css, avatar = "user", "👤"
    else:
        css, avatar = "assistant", "🩺"

    text_cls = "user-text" if role == "user" else ""
    st.markdown(
        f'<div class="chat-msg {css}">'
        f'  <div class="avatar">{avatar}</div>'
        f'  <div class="msg-body {text_cls}">{content.replace(chr(10), "<br>")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Source cards
    if role == "assistant" and sources and show_sources:
        with st.expander(f"📚 Retrieved Sources ({len(sources)})", expanded=False):
            for src in sources:
                score_disp = f"{src['score']:.3f}" if src.get("score") else "N/A"
                page_info  = f" · Page {src['page']}" if src.get("page") else ""
                st.markdown(
                    f'<div class="src-card">'
                    f'  <span class="src-title">📄 {src.get("source", "Unknown")}{page_info}</span>'
                    f'  <span class="src-score">↑ {score_disp}</span>'
                    f'  <div class="src-preview">{src.get("preview", "")}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Grounding warnings
    if role == "assistant" and grounding and not grounding.get("is_grounded") and show_grounding:
        for w in grounding.get("warnings", []):
            st.markdown(f'<div class="ground-warn">⚠️ {w}</div>', unsafe_allow_html=True)


# ─── Replay chat history ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    render_message(
        role=msg["role"],
        content=msg["content"],
        sources=msg.get("sources"),
        grounding=msg.get("grounding"),
        is_emergency=msg.get("is_emergency", False),
        is_blocked=msg.get("is_blocked", False),
    )


# ─── Quick-Action Suggestion Chips ───────────────────────────────────────────
SAMPLE_QUERIES = [
    "What are the symptoms of myocardial infarction?",
    "How is type 2 diabetes managed?",
    "Explain the mechanism of beta-blockers.",
    "What causes hypertensive crisis?",
    "Describe the pathophysiology of sepsis.",
]

if not st.session_state.messages:
    st.markdown('<div class="chip-row">' + "".join(
        f'<span class="chip" onclick="void(0)">{q}</span>' for q in SAMPLE_QUERIES
    ) + '</div>', unsafe_allow_html=True)


# ─── Input Area ──────────────────────────────────────────────────────────────
st.markdown("---")
col_input, col_btn = st.columns([5, 1])
with col_input:
    user_input = st.text_area(
        label="Ask a medical question",
        placeholder="e.g. What are the first-line treatments for community-acquired pneumonia?",
        height=90,
        key="user_input",
        label_visibility="collapsed",
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.button("Send →", use_container_width=True)

st.markdown("""
<p class="suggestions">
💡 Try: <em>"What are the signs of pulmonary embolism?"</em> &nbsp;·&nbsp;
<em>"Explain the coagulation cascade."</em> &nbsp;·&nbsp;
<em>"What is the mechanism of ACE inhibitors?"</em>
</p>
""", unsafe_allow_html=True)


# ─── Query Processing ─────────────────────────────────────────────────────────
if submit and user_input.strip():
    query = user_input.strip()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    render_message("user", query)

    # ── Safety guardrails ──────────────────────────────────────────────────
    guard = check_query(query)
    if guard.blocked:
        st.session_state.messages.append({
            "role": "assistant",
            "content": guard.override_response,
            "is_emergency": guard.is_emergency,
            "is_blocked": not guard.is_emergency,
        })
        render_message(
            "assistant",
            guard.override_response,
            is_emergency=guard.is_emergency,
            is_blocked=not guard.is_emergency,
        )
        st.session_state.total_queries += 1
        st.rerun()

    # ── RAG pipeline ───────────────────────────────────────────────────────
    with st.spinner("🔍 Searching medical knowledge base…"):
        start = time.time()
        try:
            chunks = retrieve_and_rerank(query, top_k=top_k) if use_reranker else retrieve(query, top_k=top_k)
        except Exception as e:
            chunks = []
            st.error(f"Retrieval error: {e}")

    sources = [
        {
            "source":      c.metadata.get("source", "Unknown"),
            "page":        c.metadata.get("page", ""),
            "chunk_index": c.metadata.get("chunk_index", ""),
            "score":       round(c.score, 4),
            "preview":     c.preview,
        }
        for c in chunks
    ]
    for src in sources:
        if src.get("source"):
            st.session_state.sources_used.add(src["source"])

    # ── Stream the response ────────────────────────────────────────────────
    st.markdown(
        "<p style='color:#00b4d8; font-size:0.9rem; margin-bottom:0.3rem;'>"
        "🩺 <strong>MedRAG Response:</strong></p>",
        unsafe_allow_html=True,
    )
    try:
        streamed = st.write_stream(generate_stream(query, chunks))
    except Exception as e:
        streamed = f"⚠️ Generation error: {e}"
        st.error(streamed)

    elapsed = time.time() - start
    final_text = inject_disclaimer(streamed)
    timing_note = (
        f"\n\n<small style='color:#3a5060;'>"
        f"⏱ {len(chunks)} chunks retrieved · {elapsed:.1f}s</small>"
    )

    # ── Grounding check ────────────────────────────────────────────────────
    context_texts = [c.content for c in chunks]
    grounding = check_grounding(streamed, context_texts)

    # Render grounding warnings immediately
    if not grounding.get("is_grounded") and show_grounding:
        for w in grounding.get("warnings", []):
            st.markdown(f'<div class="ground-warn">⚠️ {w}</div>', unsafe_allow_html=True)

    # Render source cards immediately
    if sources and show_sources:
        with st.expander(f"📚 Retrieved Sources ({len(sources)})", expanded=False):
            for src in sources:
                score_disp = f"{src['score']:.3f}" if src.get("score") else "N/A"
                page_info  = f" · Page {src['page']}" if src.get("page") else ""
                st.markdown(
                    f'<div class="src-card">'
                    f'  <span class="src-title">📄 {src.get("source", "Unknown")}{page_info}</span>'
                    f'  <span class="src-score">↑ {score_disp}</span>'
                    f'  <div class="src-preview">{src.get("preview", "")}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown(timing_note, unsafe_allow_html=True)

    # Save to history
    st.session_state.messages.append({
        "role":      "assistant",
        "content":   final_text,
        "sources":   sources,
        "grounding": grounding,
    })

    st.session_state.total_queries += 1
    st.rerun()
