"""
MedRAG RAG Node
----------------
LangGraph node that wraps the existing retriever + generator pipeline.

After generating the clinical answer it runs a secondary Groq LLM classifier
to decide whether a specialist consultation offer should be shown to the user.

The classifier answers: "Does this answer describe a specific medical
condition/diagnosis where professional specialist review would add value?"
"""

from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import GROQ_API_KEY, GROQ_MODEL
from src.retriever import retrieve_and_rerank, retrieve
from src.generator import generate_stream, format_context
from src.guardrails import check_grounding, inject_disclaimer
from src.agent.state import AgentState
from src.utils import get_logger

logger = get_logger(__name__)

_SPECIALIST_SYSTEM = """\
You are a triage assistant. Given a medical Q&A exchange, decide whether the
answer describes a specific condition, diagnosis, or symptom cluster where
consulting a medical specialist would genuinely benefit the patient.

Respond with exactly one word: yes  OR  no
"""


def _should_offer_consultation(question: str, answer: str) -> bool:
    """Return True if the answer warrants a specialist consultation offer."""
    try:
        import os as _os
        from dotenv import load_dotenv as _ld
        from pathlib import Path as _P
        _env = _P(__file__).resolve().parent.parent.parent / ".env"
        for _k in ("GROQ_MODEL", "GROQ_API_KEY"):
            _os.environ.pop(_k, None)
        _ld(dotenv_path=_env, override=True)
        _model = _os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        _key   = _os.getenv("GROQ_API_KEY", GROQ_API_KEY)

        llm = ChatGroq(api_key=_key, model=_model, temperature=0, max_tokens=5)
        prompt = f"Question: {question}\n\nAnswer summary: {answer[:500]}"
        result = llm.invoke([
            SystemMessage(content=_SPECIALIST_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return result.content.strip().lower().startswith("y")
    except Exception as e:
        logger.warning(f"Specialist-offer classifier failed ({e}), defaulting to False")
        return False


def rag_node(state: AgentState, top_k: int = 5, use_reranker: bool = True) -> AgentState:
    """
    LangGraph node: retrieve → rerank → generate → specialist-offer check.
    Returns updated state with rag_answer, rag_sources, rag_grounding,
    offer_consultation, and response_text.
    """
    query = state.get("user_query", "")
    logger.info(f"RAG node — query: '{query[:80]}'")

    # ── 1. Retrieve chunks ──────────────────────────────────────────────────
    try:
        if use_reranker:
            chunks = retrieve_and_rerank(query, top_k=top_k)
        else:
            chunks = retrieve(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return {
            **state,
            "rag_answer": "⚠️ Could not retrieve medical context. Please check your MongoDB connection.",
            "rag_sources": [],
            "rag_grounding": {"is_grounded": False, "warnings": [str(e)]},
            "offer_consultation": False,
            "response_text": "⚠️ Retrieval error. Please check your connection.",
            "next_node": "end",
        }

    # ── 2. Generate (non-streaming path for graph; UI does streaming separately) ──
    context = format_context(chunks)
    from src.generator import _build_messages, _get_llm

    try:
        llm = _get_llm()
        messages = _build_messages(context, query)
        result = llm.invoke(messages)
        raw_answer = result.content
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raw_answer = f"⚠️ Generation error: {e}"

    # ── 3. Grounding check ──────────────────────────────────────────────────
    context_texts = [c.content for c in chunks]
    grounding = check_grounding(raw_answer, context_texts)

    # ── 4. Disclaimer ───────────────────────────────────────────────────────
    final_answer = inject_disclaimer(raw_answer)

    # ── 5. Build sources list ───────────────────────────────────────────────
    sources = [
        {
            "source": c.metadata.get("source", "Unknown"),
            "page": c.metadata.get("page", ""),
            "chunk_index": c.metadata.get("chunk_index", ""),
            "score": round(c.score, 4),
            "preview": c.preview,
        }
        for c in chunks
    ]

    # ── 6. Should we offer a specialist consultation? ───────────────────────
    offer = _should_offer_consultation(query, raw_answer)
    logger.info(f"Specialist offer: {offer}")

    return {
        **state,
        "rag_answer": final_answer,
        "rag_sources": sources,
        "rag_grounding": grounding,
        "offer_consultation": offer,
        "response_text": final_answer,
        "next_node": "end",
    }
