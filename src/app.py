"""
MedRAG — Medical AI Assistant
------------------------------
LLM + RAG pipeline for medical students and healthcare professionals.
Powered by Groq LLM · MongoDB Atlas Vector Search · HuggingFace Embeddings.

Tabs:
  🩺 AI Assistant   — Chat interface with RAG pipeline
  🗄️ Database       — CRUD operations + Aggregation Pipeline analytics

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
    validate_config, get_mongo_client,
)
from src.guardrails import check_query
from src.retriever import retrieve_and_rerank, retrieve
from src.generator import generate_stream
from src.guardrails import check_grounding, inject_disclaimer
from src.database import (
    # CRUD
    create_chunk, read_chunks, get_all_sources,
    get_chunk_by_id, update_chunk_meta, update_chunk_text,
    delete_chunk, delete_by_source, delete_all_chunks,
    # Aggregation
    agg_knowledge_base_summary, agg_chunks_per_source,
    agg_filetype_distribution, agg_chunk_length_buckets,
    agg_stale_embeddings, agg_query_history, agg_query_stats,
    # Query logging
    log_query,
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedRAG — Medical AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "MedRAG: Evidence-based Medical AI for students and professionals. Groq + MongoDB Atlas + RAG.",
    },
)

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #060d18 0%, #0a1628 55%, #080f1e 100%);
    min-height: 100vh;
}
.main .block-container { padding: 2rem 2rem 5rem 2rem; max-width: 1200px; }

/* Hero */
.hero {
    text-align: center; padding: 2.5rem 1.5rem 1.8rem;
    background: linear-gradient(135deg, rgba(0,180,216,0.07), rgba(100,60,220,0.07));
    border: 1px solid rgba(0,180,216,0.14); border-radius: 22px;
    margin-bottom: 2rem; position: relative; overflow: hidden;
}
.hero::before {
    content:''; position:absolute; inset:0;
    background: radial-gradient(ellipse at 50% 0%, rgba(0,180,216,0.08) 0%, transparent 70%);
    pointer-events:none;
}
.hero h1 {
    font-size: 2.8rem; font-weight: 700; margin: 0 0 0.3rem;
    background: linear-gradient(90deg, #00c8f0, #8b5cf6, #00c8f0);
    background-size: 200%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 5s linear infinite;
}
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }
.hero .tagline { color: #7a93b0; font-size: 1rem; margin: 0 0 1.1rem; }
.badge {
    display: inline-block;
    background: rgba(0,180,216,0.1); border: 1px solid rgba(0,180,216,0.25);
    color: #00b4d8; border-radius: 20px; padding: 3px 13px;
    font-size: 0.73rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; margin: 0 4px;
}
.badge.purple { background: rgba(139,92,246,0.1); border-color: rgba(139,92,246,0.25); color: #a78bfa; }
.badge.green  { background: rgba(16,185,129,0.1);  border-color: rgba(16,185,129,0.25);  color: #34d399; }
.badge.amber  { background: rgba(245,158,11,0.1);  border-color: rgba(245,158,11,0.25);  color: #fbbf24; }

/* Chat */
.chat-msg {
    display:flex; gap:14px; padding:1.1rem 1.3rem;
    border-radius:18px; margin-bottom:1rem; animation: fadeUp 0.3s ease;
}
@keyframes fadeUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
.chat-msg.user {
    background: linear-gradient(135deg,rgba(0,180,216,0.09),rgba(0,180,216,0.04));
    border:1px solid rgba(0,180,216,0.18); flex-direction:row-reverse;
}
.chat-msg.assistant {
    background: linear-gradient(135deg,rgba(139,92,246,0.09),rgba(139,92,246,0.03));
    border:1px solid rgba(139,92,246,0.18);
}
.chat-msg.emergency {
    background: linear-gradient(135deg,rgba(239,68,68,0.12),rgba(239,68,68,0.04));
    border:1px solid rgba(239,68,68,0.35);
}
.chat-msg.blocked {
    background: linear-gradient(135deg,rgba(245,158,11,0.09),rgba(245,158,11,0.03));
    border:1px solid rgba(245,158,11,0.25);
}
.avatar { font-size:1.5rem; flex-shrink:0; width:38px; text-align:center; padding-top:2px; }
.msg-body { flex:1; color:#cdd8e8; line-height:1.7; font-size:0.96rem; }
.msg-body.user-text { text-align:right; color:#9ab8d4; }

/* Source cards */
.src-card {
    background: rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.07);
    border-radius:10px; padding:0.7rem 1rem; margin-bottom:0.5rem;
    font-size:0.84rem; color:#8b9ab0; transition:border-color 0.2s, background 0.2s;
}
.src-card:hover { border-color:rgba(0,180,216,0.35); background:rgba(0,180,216,0.04); }
.src-title { color:#00b4d8; font-weight:600; font-size:0.86rem; }
.src-score {
    float:right; background:rgba(0,180,216,0.12); color:#00b4d8;
    padding:1px 9px; border-radius:10px; font-size:0.74rem;
    font-family:'JetBrains Mono',monospace;
}
.src-preview { margin-top:0.35rem; font-style:italic; line-height:1.5; }
.ground-warn {
    background:rgba(245,158,11,0.07); border:1px solid rgba(245,158,11,0.25);
    border-radius:8px; padding:0.45rem 0.8rem; color:#fbbf24;
    font-size:0.81rem; margin-top:0.5rem;
}

/* DB section cards */
.db-section {
    background: rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.07);
    border-radius:16px; padding:1.4rem 1.6rem; margin-bottom:1.5rem;
}
.db-section h3 { color:#00b4d8; font-size:1rem; margin:0 0 1rem; }

/* Stat cards */
.stat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.5rem; }
.stat-card {
    background: linear-gradient(135deg,rgba(0,180,216,0.07),rgba(0,180,216,0.03));
    border:1px solid rgba(0,180,216,0.18); border-radius:14px;
    padding:1rem 1.2rem; text-align:center;
}
.stat-val { font-size:2rem; font-weight:700; color:#00b4d8; font-family:'JetBrains Mono',monospace; }
.stat-label { font-size:0.78rem; color:#7a93b0; margin-top:0.2rem; }

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.035); border:1px solid rgba(255,255,255,0.07);
    border-radius:12px; padding:0.7rem 1rem;
}
[data-testid="stMetricLabel"] { color:#7a93b0 !important; font-size:0.78rem !important; }
[data-testid="stMetricValue"] { color:#00b4d8 !important; font-size:1.4rem !important; font-weight:700 !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0a1322 0%,#060d18 100%);
    border-right:1px solid rgba(255,255,255,0.055);
}
[data-testid="stSidebar"] .stMarkdown { color:#7a93b0; }

/* Inputs */
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.045) !important;
    border:1px solid rgba(0,180,216,0.22) !important;
    border-radius:12px !important; color:#cdd8e8 !important;
    font-family:'Inter',sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color:rgba(0,180,216,0.55) !important;
    box-shadow:0 0 0 3px rgba(0,180,216,0.1) !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.045) !important;
    border:1px solid rgba(0,180,216,0.22) !important;
    border-radius:10px !important; color:#cdd8e8 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg,#00b4d8,#7c3aed);
    color:white; border:none; border-radius:10px;
    font-weight:600; padding:0.5rem 1.4rem;
    transition:opacity 0.2s, transform 0.15s;
}
.stButton > button:hover { opacity:0.85; transform:translateY(-1px); }
.stButton > button[kind="secondary"] {
    background: rgba(239,68,68,0.15) !important;
    border:1px solid rgba(239,68,68,0.3) !important;
    color:#f87171 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px; padding: 4px; gap: 4px;
    border: 1px solid rgba(255,255,255,0.07);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px; color: #7a93b0 !important;
    font-weight: 500; padding: 0.5rem 1.2rem;
    transition: background 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,rgba(0,180,216,0.2),rgba(124,58,237,0.2)) !important;
    color: #fff !important;
}

/* Tables */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Expanders */
.streamlit-expanderHeader { color: #00b4d8 !important; font-size: 0.9rem !important; }

hr { border-color:rgba(255,255,255,0.06) !important; }
.stSpinner > div { border-top-color:#00b4d8 !important; }
.suggestions { color:#4a6070; font-size:0.82rem; margin-top:0.4rem; }
.suggestions em { color:#5a7a90; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages": [],
        "total_queries": 0,
        "sources_used": set(),
        "uploaded_filenames": set(),
        "db_page": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ RAG Settings")
    top_k = st.slider("Context chunks", 1, 10, TOP_K, 1,
                      help="Number of chunks retrieved per query.")
    show_sources   = st.toggle("Show source excerpts",           value=True)
    show_grounding = st.toggle("Show grounding warnings",        value=True)
    use_reranker   = st.toggle("Enable cross-encoder reranking", value=True)

    st.markdown("---")
    st.markdown("## 📥 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT, CSV, JSON",
        type=["pdf","txt","csv","json","jsonl"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        if st.button("🚀 Ingest to MongoDB", use_container_width=True):
            with st.spinner("Embedding & storing in MongoDB Atlas…"):
                docs_dir = Path(DOCS_DIR)
                docs_dir.mkdir(parents=True, exist_ok=True)
                for u in uploaded_files:
                    fp = docs_dir / u.name
                    with open(fp, "wb") as f:
                        f.write(u.getbuffer())
                    st.session_state.uploaded_filenames.add(u.name)
                from src.ingestion import run_ingestion
                run_ingestion(docs_dir=str(docs_dir))
            st.success(f"✅ Ingested {len(uploaded_files)} file(s)!")

    if st.session_state.uploaded_filenames:
        st.markdown(
            f"<small style='color:#00b4d8;'>Indexed: {', '.join(st.session_state.uploaded_filenames)}</small>",
            unsafe_allow_html=True,
        )
        if st.button("🧹 Remove Session Docs", use_container_width=True):
            from src.ingestion import delete_uploaded_documents
            delete_uploaded_documents(list(st.session_state.uploaded_filenames))
            st.session_state.uploaded_filenames = set()
            st.rerun()

    st.markdown("---")
    st.markdown("## 📊 Session Stats")
    c1, c2 = st.columns(2)
    with c1: st.metric("Queries", st.session_state.total_queries)
    with c2: st.metric("Sources", len(st.session_state.sources_used))

    st.markdown("---")
    st.markdown("## 🤖 Stack")
    st.markdown(f"""
| | |
|---|---|
| **LLM** | `{GROQ_MODEL}` |
| **Embeddings** | MiniLM-L6-v2 |
| **Vector DB** | MongoDB Atlas |
| **Reranker** | ms-marco-MiniLM |
""")
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
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
    st.markdown("<small style='color:#3a5060;'>MedRAG v3.0 · Educational use only</small>",
                unsafe_allow_html=True)


# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🩺 MedRAG</h1>
    <p class="tagline">AI-Powered Medical Knowledge Assistant for Students &amp; Experts</p>
    <span class="badge">Groq LLM</span>
    <span class="badge purple">RAG Pipeline</span>
    <span class="badge green">MongoDB Atlas</span>
    <span class="badge amber">CRUD + Aggregation</span>
</div>
""", unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_chat, tab_db = st.tabs(["🩺 AI Assistant", "🗄️ Database Manager"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:

    # Message renderer
    def render_message(role, content, sources=None, grounding=None,
                       is_emergency=False, is_blocked=False):
        if is_emergency:   css, avatar = "emergency", "🚨"
        elif is_blocked:   css, avatar = "blocked",   "⚠️"
        elif role=="user": css, avatar = "user",      "👤"
        else:              css, avatar = "assistant",  "🩺"
        text_cls = "user-text" if role == "user" else ""
        st.markdown(
            f'<div class="chat-msg {css}">'
            f'<div class="avatar">{avatar}</div>'
            f'<div class="msg-body {text_cls}">{content.replace(chr(10),"<br>")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if role == "assistant" and sources and show_sources:
            with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                for src in sources:
                    sd = f"{src['score']:.3f}" if src.get("score") else "N/A"
                    pi = f" · Page {src['page']}" if src.get("page") else ""
                    st.markdown(
                        f'<div class="src-card">'
                        f'<span class="src-title">📄 {src.get("source","Unknown")}{pi}</span>'
                        f'<span class="src-score">↑ {sd}</span>'
                        f'<div class="src-preview">{src.get("preview","")}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        if role == "assistant" and grounding and not grounding.get("is_grounded") and show_grounding:
            for w in grounding.get("warnings", []):
                st.markdown(f'<div class="ground-warn">⚠️ {w}</div>', unsafe_allow_html=True)

    # Replay history
    for msg in st.session_state.messages:
        render_message(
            role=msg["role"], content=msg["content"],
            sources=msg.get("sources"), grounding=msg.get("grounding"),
            is_emergency=msg.get("is_emergency", False),
            is_blocked=msg.get("is_blocked", False),
        )

    # Welcome chips
    SAMPLE_QUERIES = [
        "What are the symptoms of myocardial infarction?",
        "How is type 2 diabetes managed?",
        "Explain the mechanism of beta-blockers.",
        "What causes hypertensive crisis?",
        "Describe the pathophysiology of sepsis.",
    ]
    if not st.session_state.messages:
        st.markdown('<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:1rem;">' + "".join(
            f'<span style="background:rgba(0,180,216,0.07);border:1px solid rgba(0,180,216,0.2);'
            f'color:#00b4d8;border-radius:20px;padding:4px 14px;font-size:0.8rem;">{q}</span>'
            for q in SAMPLE_QUERIES
        ) + '</div>', unsafe_allow_html=True)

    st.markdown("---")
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        user_input = st.text_area(
            "Ask a medical question",
            placeholder="e.g. What are the first-line treatments for community-acquired pneumonia?",
            height=90, key="user_input", label_visibility="collapsed",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.button("Send →", use_container_width=True)

    st.markdown("""
<p class="suggestions">
💡 Try: <em>"Explain the coagulation cascade."</em> &nbsp;·&nbsp;
<em>"Signs of pulmonary embolism?"</em> &nbsp;·&nbsp;
<em>"Mechanism of ACE inhibitors?"</em>
</p>""", unsafe_allow_html=True)

    # Query processing
    if submit and user_input.strip():
        query = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": query})
        render_message("user", query)

        guard = check_query(query)
        if guard.blocked:
            st.session_state.messages.append({
                "role": "assistant", "content": guard.override_response,
                "is_emergency": guard.is_emergency,
                "is_blocked": not guard.is_emergency,
            })
            render_message("assistant", guard.override_response,
                           is_emergency=guard.is_emergency,
                           is_blocked=not guard.is_emergency)
            st.session_state.total_queries += 1
            st.rerun()

        with st.spinner("🔍 Searching knowledge base…"):
            start = time.time()
            try:
                chunks = retrieve_and_rerank(query, top_k=top_k) if use_reranker else retrieve(query, top_k=top_k)
            except Exception as e:
                chunks = []
                st.error(f"Retrieval error: {e}")

        sources = [
            {"source": c.metadata.get("source","Unknown"),
             "page": c.metadata.get("page",""),
             "chunk_index": c.metadata.get("chunk_index",""),
             "score": round(c.score, 4), "preview": c.preview}
            for c in chunks
        ]
        for s in sources:
            if s.get("source"):
                st.session_state.sources_used.add(s["source"])

        st.markdown(
            "<p style='color:#00b4d8;font-size:0.9rem;margin-bottom:0.3rem;'>"
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
        timing = (f"\n\n<small style='color:#3a5060;'>"
                  f"⏱ {len(chunks)} chunks · {elapsed:.1f}s</small>")

        grounding = check_grounding(streamed, [c.content for c in chunks])
        if not grounding.get("is_grounded") and show_grounding:
            for w in grounding.get("warnings", []):
                st.markdown(f'<div class="ground-warn">⚠️ {w}</div>', unsafe_allow_html=True)

        if sources and show_sources:
            with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                for src in sources:
                    sd = f"{src['score']:.3f}" if src.get("score") else "N/A"
                    pi = f" · Page {src['page']}" if src.get("page") else ""
                    st.markdown(
                        f'<div class="src-card">'
                        f'<span class="src-title">📄 {src.get("source","Unknown")}{pi}</span>'
                        f'<span class="src-score">↑ {sd}</span>'
                        f'<div class="src-preview">{src.get("preview","")}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        st.markdown(timing, unsafe_allow_html=True)

        # Log the query to MongoDB
        try:
            log_query(query, len(chunks), elapsed)
        except Exception:
            pass

        st.session_state.messages.append({
            "role": "assistant", "content": final_text,
            "sources": sources, "grounding": grounding,
        })
        st.session_state.total_queries += 1
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — DATABASE MANAGER
# ══════════════════════════════════════════════════════════════════════════════
with tab_db:

    st.markdown("## 🗄️ MongoDB Knowledge Base Manager")
    st.markdown(
        "<small style='color:#7a93b0;'>Perform CRUD operations and run Aggregation Pipelines "
        "on the MedRAG MongoDB Atlas collection.</small>",
        unsafe_allow_html=True,
    )

    # ── Sub-tabs inside Database Manager ──────────────────────────────────────
    db_overview, db_crud, db_agg, db_logs = st.tabs([
        "📊 Overview", "✏️ CRUD Operations",
        "🔬 Aggregation Pipelines", "📜 Query Logs",
    ])

    # ────────────────────────────────────────────────────────────────────────
    #  OVERVIEW
    # ────────────────────────────────────────────────────────────────────────
    with db_overview:
        st.markdown("### 📊 Knowledge Base Summary")
        if st.button("🔄 Refresh Stats", key="refresh_overview"):
            st.cache_data.clear()

        with st.spinner("Fetching aggregation stats…"):
            try:
                summary = agg_knowledge_base_summary()
                per_source = agg_chunks_per_source()
                filetypes = agg_filetype_distribution()
                buckets = agg_chunk_length_buckets()
                stale = agg_stale_embeddings()
            except Exception as e:
                st.error(f"Could not connect to MongoDB: {e}")
                st.stop()

        # Stat cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Chunks",    summary.get("total_chunks", 0))
        with c2:
            st.metric("Unique Sources",  summary.get("unique_sources", 0))
        with c3:
            st.metric("Avg Chunk Length", f"{summary.get('avg_text_length',0):,} chars")
        with c4:
            st.metric("Total Characters", f"{summary.get('total_characters',0):,}")

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 📄 Chunks per Source")
            if per_source:
                import pandas as pd
                df_src = pd.DataFrame(per_source)
                df_src.columns = ["Source", "Chunks", "Avg Length (chars)", "File Type"]
                st.dataframe(df_src, use_container_width=True, hide_index=True)
            else:
                st.info("No documents ingested yet.")

        with col_b:
            st.markdown("#### 🗂️ File Type Distribution")
            if filetypes:
                import pandas as pd
                df_ft = pd.DataFrame(filetypes)
                df_ft.columns = ["File Type", "Chunk Count", "Percentage (%)"]
                st.dataframe(df_ft, use_container_width=True, hide_index=True)

                st.markdown("#### 📏 Chunk Length Buckets")
                if buckets:
                    df_bk = pd.DataFrame(buckets)
                    df_bk.columns = ["Bucket", "Count"]
                    st.dataframe(df_bk, use_container_width=True, hide_index=True)
            else:
                st.info("No documents ingested yet.")

        st.markdown("---")
        st.markdown("#### ⚠️ Embedding Health")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.metric("✅ Healthy Embeddings", stale.get("ok_count", 0))
        with sc2:
            st.metric("⚠️ Stale Embeddings",   stale.get("stale_count", 0),
                      delta=None if stale.get("stale_count",0)==0 else f"{stale.get('stale_count',0)} need re-embed",
                      delta_color="inverse")


    # ────────────────────────────────────────────────────────────────────────
    #  CRUD OPERATIONS
    # ────────────────────────────────────────────────────────────────────────
    with db_crud:

        crud_create, crud_read, crud_update, crud_delete = st.tabs([
            "➕ Create", "🔍 Read", "✏️ Update", "🗑️ Delete",
        ])

        # ── CREATE ──────────────────────────────────────────────────────────
        with crud_create:
            st.markdown("### ➕ Insert a Knowledge Chunk")
            st.info("Manually insert a medical knowledge entry into MongoDB. "
                    "Note: entries inserted here do NOT have a vector embedding "
                    "and won't appear in semantic search. Use the Upload sidebar for "
                    "fully-embedded ingestion.", icon="ℹ️")

            with st.form("create_form"):
                c_text     = st.text_area("📝 Chunk Text *", height=150,
                                          placeholder="Enter medical knowledge text…")
                c_source   = st.text_input("📄 Source Name *",
                                           placeholder="e.g. cardiology_notes.txt")
                c_filetype = st.selectbox("📁 File Type", ["manual","pdf","txt","csv","json"])
                c_page     = st.number_input("📃 Page Number (optional)", min_value=0, value=0)
                c_tags     = st.text_input("🏷️ Tags (comma-separated, optional)",
                                           placeholder="e.g. cardiology, hypertension")
                submitted = st.form_submit_button("💾 Insert Chunk", use_container_width=True)

                if submitted:
                    if not c_text.strip() or not c_source.strip():
                        st.error("Text and Source are required.")
                    else:
                        extra = {}
                        if c_tags.strip():
                            extra["tags"] = [t.strip() for t in c_tags.split(",") if t.strip()]
                        with st.spinner("Inserting…"):
                            inserted_id = create_chunk(
                                text=c_text.strip(),
                                source=c_source.strip(),
                                file_type=c_filetype,
                                page=int(c_page) if c_page else None,
                                extra_meta=extra if extra else None,
                            )
                        st.success(f"✅ Inserted! Document `_id`: `{inserted_id}`")

        # ── READ ─────────────────────────────────────────────────────────────
        with crud_read:
            st.markdown("### 🔍 Browse Knowledge Chunks")

            col_f1, col_f2 = st.columns([2, 2])
            with col_f1:
                try:
                    all_sources = ["(All)"] + get_all_sources()
                except Exception:
                    all_sources = ["(All)"]
                src_filter = st.selectbox("Filter by Source", all_sources, key="read_src")
            with col_f2:
                kw_filter = st.text_input("🔎 Keyword in text", placeholder="e.g. insulin",
                                          key="read_kw")

            page_size = 10
            page_num  = st.session_state.get("db_page", 0)

            with st.spinner("Loading chunks…"):
                try:
                    docs, total = read_chunks(
                        source_filter=None if src_filter == "(All)" else src_filter,
                        keyword=kw_filter.strip() or None,
                        skip=page_num * page_size,
                        limit=page_size,
                    )
                except Exception as e:
                    docs, total = [], 0
                    st.error(f"Read error: {e}")

            st.markdown(
                f"<small style='color:#7a93b0;'>Showing {len(docs)} of **{total}** "
                f"matching chunks (page {page_num+1})</small>",
                unsafe_allow_html=True,
            )

            # Pagination
            nav_l, nav_info, nav_r = st.columns([1, 2, 1])
            with nav_l:
                if st.button("◀ Prev", disabled=page_num == 0, key="pg_prev"):
                    st.session_state.db_page = max(0, page_num - 1)
                    st.rerun()
            with nav_info:
                total_pages = max(1, (total + page_size - 1) // page_size)
                st.markdown(
                    f"<p style='text-align:center;color:#7a93b0;margin:0.5rem 0;'>"
                    f"Page {page_num+1} / {total_pages}</p>",
                    unsafe_allow_html=True,
                )
            with nav_r:
                if st.button("Next ▶", disabled=(page_num + 1) >= total_pages, key="pg_next"):
                    st.session_state.db_page = page_num + 1
                    st.rerun()

            for doc in docs:
                with st.expander(
                    f"📄 {doc.get('source','?')}  ·  chunk #{doc.get('chunk_index','')}  "
                    f"·  `{doc['_id']}`",
                    expanded=False,
                ):
                    text_preview = doc.get("text", "")[:600]
                    st.markdown(
                        f"<div style='background:rgba(0,180,216,0.04);border:1px solid rgba(0,180,216,0.12);"
                        f"border-radius:8px;padding:0.8rem 1rem;font-size:0.87rem;color:#cdd8e8;line-height:1.6;'>"
                        f"{text_preview}{'…' if len(doc.get('text',''))>600 else ''}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    meta_cols = st.columns(4)
                    meta_cols[0].markdown(f"**Source:** {doc.get('source','?')}")
                    meta_cols[1].markdown(f"**File Type:** {doc.get('file_type','?')}")
                    meta_cols[2].markdown(f"**Page:** {doc.get('page','–')}")
                    meta_cols[3].markdown(f"**Length:** {len(doc.get('text',''))} chars")
                    if doc.get("tags"):
                        st.markdown(f"**Tags:** {', '.join(doc['tags'])}")
                    if doc.get("created_at"):
                        st.markdown(f"<small style='color:#4a6070;'>Created: {doc['created_at']}</small>",
                                    unsafe_allow_html=True)

        # ── UPDATE ───────────────────────────────────────────────────────────
        with crud_update:
            st.markdown("### ✏️ Update a Chunk")
            st.info("Paste a chunk `_id` from the Read tab to update its metadata or text.", icon="ℹ️")

            upd_id = st.text_input("🔑 Chunk `_id` to update", placeholder="65f3a0b2e4b0a1c2d3e4f5a6",
                                   key="upd_id")
            if upd_id.strip():
                doc = get_chunk_by_id(upd_id.strip())
                if doc:
                    st.markdown(
                        f"<div style='background:rgba(0,180,216,0.04);border:1px solid rgba(0,180,216,0.12);"
                        f"border-radius:8px;padding:0.8rem 1rem;font-size:0.87rem;color:#8b9ab0;'>"
                        f"📄 <strong>{doc.get('source','?')}</strong> · "
                        f"chunk #{doc.get('chunk_index','?')} · {len(doc.get('text',''))} chars"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("#### Update Metadata Fields")
                    with st.form("update_meta_form"):
                        new_source = st.text_input("📄 Source name",
                                                   value=doc.get("source",""), key="u_src")
                        new_tags   = st.text_input("🏷️ Tags (comma-separated)",
                                                   value=", ".join(doc.get("tags",[])), key="u_tags")
                        upd_submitted = st.form_submit_button("💾 Update Metadata")
                        if upd_submitted:
                            updates: dict = {}
                            if new_source.strip():
                                updates["source"] = new_source.strip()
                            if new_tags.strip():
                                updates["tags"] = [t.strip() for t in new_tags.split(",") if t.strip()]
                            if updates:
                                ok = update_chunk_meta(upd_id.strip(), updates)
                                if ok:
                                    st.success("✅ Metadata updated!")
                                else:
                                    st.warning("No changes made.")
                            else:
                                st.warning("Nothing to update.")

                    st.markdown("#### Update Chunk Text")
                    st.warning("⚠️ Updating text makes the vector embedding stale. "
                               "Re-ingest the source file to refresh embeddings.", icon="⚠️")
                    with st.form("update_text_form"):
                        new_text = st.text_area("📝 New text",
                                                value=doc.get("text",""), height=200, key="u_text")
                        txt_submitted = st.form_submit_button("💾 Update Text")
                        if txt_submitted:
                            if new_text.strip():
                                ok = update_chunk_text(upd_id.strip(), new_text.strip())
                                if ok:
                                    st.success("✅ Text updated! Embedding marked as stale.")
                                else:
                                    st.error("Update failed.")
                            else:
                                st.error("Text cannot be empty.")
                else:
                    st.error(f"No chunk found with id: `{upd_id}`")

        # ── DELETE ───────────────────────────────────────────────────────────
        with crud_delete:
            st.markdown("### 🗑️ Delete Chunks")

            st.markdown("#### Delete by Chunk `_id`")
            with st.form("delete_one_form"):
                del_id = st.text_input("🔑 Chunk `_id`",
                                       placeholder="65f3a0b2e4b0a1c2d3e4f5a6", key="del_id")
                del_one = st.form_submit_button("🗑️ Delete This Chunk")
                if del_one and del_id.strip():
                    ok = delete_chunk(del_id.strip())
                    if ok:
                        st.success(f"✅ Deleted chunk `{del_id}`")
                    else:
                        st.error("Chunk not found.")

            st.markdown("---")
            st.markdown("#### Delete All Chunks by Source")
            try:
                sources_list = get_all_sources()
            except Exception:
                sources_list = []

            if sources_list:
                with st.form("delete_source_form"):
                    del_src = st.selectbox("📄 Select Source", sources_list, key="del_src")
                    del_src_btn = st.form_submit_button("🗑️ Delete All from This Source")
                    if del_src_btn:
                        count = delete_by_source(del_src)
                        st.success(f"✅ Deleted **{count}** chunks from `{del_src}`.")
            else:
                st.info("No sources in the database yet.")

            st.markdown("---")
            st.markdown("#### ⚠️ Nuclear Option — Delete All Chunks")
            st.error(
                "This will permanently delete **every chunk** in the medical knowledge base. "
                "You will need to re-ingest all documents.",
                icon="🚨",
            )
            col_check, col_nuke = st.columns([3, 1])
            with col_check:
                confirm = st.checkbox("I understand this is irreversible", key="nuke_confirm")
            with col_nuke:
                if st.button("💥 Delete ALL", disabled=not confirm, key="nuke_btn"):
                    count = delete_all_chunks()
                    st.success(f"🧹 Deleted all {count} chunks.")
                    st.session_state.sources_used = set()
                    st.session_state.uploaded_filenames = set()
                    st.rerun()


    # ────────────────────────────────────────────────────────────────────────
    #  AGGREGATION PIPELINES
    # ────────────────────────────────────────────────────────────────────────
    with db_agg:
        st.markdown("### 🔬 MongoDB Aggregation Pipelines")
        st.markdown(
            "<small style='color:#7a93b0;'>Live aggregation queries running directly "
            "on MongoDB Atlas — no pre-computed caches.</small>",
            unsafe_allow_html=True,
        )

        if st.button("🔄 Run All Pipelines", key="run_agg"):
            st.cache_data.clear()

        import pandas as pd

        with st.spinner("Running aggregation pipelines…"):
            try:
                per_source  = agg_chunks_per_source()
                filetypes   = agg_filetype_distribution()
                buckets     = agg_chunk_length_buckets()
                qstats      = agg_query_stats()
            except Exception as e:
                st.error(f"Aggregation error: {e}")
                st.stop()

        # Pipeline 1
        with st.expander("📌 Pipeline 1 — Chunks Grouped by Source Document", expanded=True):
            st.code("""db.medical_chunks.aggregate([
  { $group: {
      _id: "$source",
      chunk_count: { $sum: 1 },
      avg_length:  { $avg: { $strLenCP: "$text" } },
      file_type:   { $first: "$file_type" }
  }},
  { $sort: { chunk_count: -1 } },
  { $project: { _id:0, source:"$_id", chunk_count:1, avg_length:{$round:["$avg_length",0]}, file_type:1 }}
])""", language="javascript")
            if per_source:
                df = pd.DataFrame(per_source)
                df.columns = ["Source", "Chunk Count", "Avg Length (chars)", "File Type"]
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.bar_chart(df.set_index("Source")["Chunk Count"])
            else:
                st.info("No data yet — ingest some documents first.")

        # Pipeline 2
        with st.expander("📌 Pipeline 2 — File Type Distribution with Percentages"):
            st.code("""db.medical_chunks.aggregate([
  { $group: { _id: "$file_type", count: { $sum: 1 } }},
  { $group: { _id: null, types: { $push: { file_type:"$_id", count:"$count" } }, total:{ $sum:"$count" }}},
  { $unwind: "$types" },
  { $project: { _id:0, file_type:"$types.file_type", count:"$types.count",
      pct: { $round:[{ $multiply:[{ $divide:["$types.count","$total"]},100]},1] }
  }},
  { $sort: { count: -1 } }
])""", language="javascript")
            if filetypes:
                df_ft = pd.DataFrame(filetypes)
                df_ft.columns = ["File Type", "Count", "Percentage (%)"]
                st.dataframe(df_ft, use_container_width=True, hide_index=True)
            else:
                st.info("No data yet.")

        # Pipeline 3
        with st.expander("📌 Pipeline 3 — Chunk Length Distribution (Bucketed)"):
            st.code("""db.medical_chunks.aggregate([
  { $bucket: {
      groupBy: { $strLenCP: "$text" },
      boundaries: [0, 200, 600, 1000, 10000],
      default: "other",
      output: { count: { $sum: 1 } }
  }},
  // + $project to map boundaries to labels
])""", language="javascript")
            if buckets:
                df_bk = pd.DataFrame(buckets)
                df_bk.columns = ["Bucket", "Count"]
                st.dataframe(df_bk, use_container_width=True, hide_index=True)
                st.bar_chart(df_bk.set_index("Bucket")["Count"])
            else:
                st.info("No data yet.")

        # Pipeline 4 — Query Performance
        with st.expander("📌 Pipeline 4 — Query Performance Stats (query_logs collection)"):
            st.code("""db.query_logs.aggregate([
  { $group: {
      _id: null,
      total_queries: { $sum: 1 },
      avg_elapsed_s: { $avg: "$elapsed_s" },
      max_elapsed_s: { $max: "$elapsed_s" },
      avg_chunks:    { $avg: "$chunks_retrieved" }
  }},
  { $project: { _id:0, total_queries:1,
      avg_elapsed_s:{ $round:["$avg_elapsed_s",2] },
      max_elapsed_s:{ $round:["$max_elapsed_s",2] },
      avg_chunks:   { $round:["$avg_chunks",1] }
  }}
])""", language="javascript")
            qs1, qs2, qs3, qs4 = st.columns(4)
            qs1.metric("Total Queries",    qstats.get("total_queries", 0))
            qs2.metric("Avg Response (s)", qstats.get("avg_elapsed_s", 0))
            qs3.metric("Max Response (s)", qstats.get("max_elapsed_s", 0))
            qs4.metric("Avg Chunks/Query", qstats.get("avg_chunks", 0))

        # Pipeline 5 — Stale Embeddings
        with st.expander("📌 Pipeline 5 — Stale Embedding Detection"):
            st.code("""db.medical_chunks.aggregate([
  { $group: {
      _id: null,
      stale_count: { $sum: { $cond:[{ $eq:["$embedding_stale",true]},1,0] }},
      ok_count:    { $sum: { $cond:[{ $ne:["$embedding_stale",true]},1,0] }}
  }},
  { $project: { _id:0, stale_count:1, ok_count:1 } }
])""", language="javascript")
            try:
                stale = agg_stale_embeddings()
                se1, se2 = st.columns(2)
                se1.metric("✅ Healthy",  stale.get("ok_count", 0))
                se2.metric("⚠️ Stale",   stale.get("stale_count", 0))
            except Exception as e:
                st.error(f"Error: {e}")


    # ────────────────────────────────────────────────────────────────────────
    #  QUERY LOGS
    # ────────────────────────────────────────────────────────────────────────
    with db_logs:
        st.markdown("### 📜 Query History")
        st.markdown(
            "<small style='color:#7a93b0;'>Every query through the AI Assistant is logged "
            "to MongoDB's <code>query_logs</code> collection.</small>",
            unsafe_allow_html=True,
        )

        if st.button("🔄 Refresh Logs", key="refresh_logs"):
            pass

        try:
            logs = agg_query_history(limit=30)
            qstats = agg_query_stats()
        except Exception as e:
            logs = []
            qstats = {}
            st.error(f"Could not fetch logs: {e}")

        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Total Queries",    qstats.get("total_queries", 0))
        l2.metric("Avg Time (s)",     qstats.get("avg_elapsed_s", 0))
        l3.metric("Max Time (s)",     qstats.get("max_elapsed_s", 0))
        l4.metric("Avg Chunks/Query", qstats.get("avg_chunks", 0))

        st.markdown("---")
        if logs:
            import pandas as pd
            df_logs = pd.DataFrame(logs)
            if "timestamp" in df_logs.columns:
                df_logs = df_logs[["timestamp", "query", "chunks_retrieved", "elapsed_s"]]
                df_logs.columns = ["Timestamp (UTC)", "Query", "Chunks Retrieved", "Time (s)"]
            st.dataframe(df_logs, use_container_width=True, hide_index=True)
        else:
            st.info("No queries logged yet. Ask a question in the AI Assistant tab to start logging.")
