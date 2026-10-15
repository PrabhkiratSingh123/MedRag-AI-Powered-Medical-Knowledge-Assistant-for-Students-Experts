"""
MedRAG Response Generator
--------------------------
Constructs a grounded prompt from the retrieved medical context and
generates a response using Groq Cloud API (llama-3.3-70b-versatile by default).

Steps:
  1. Format retrieved chunks into a structured context block.
  2. Build a medically-focused system + user prompt.
  3. Call Groq API to generate a response (streaming supported).
  4. Apply grounding checks and inject the medical disclaimer.

Usage (standalone test):
    python -m src.generator "What are the risk factors for type 2 diabetes?"
"""

import sys

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
)
from src.guardrails import check_grounding, inject_disclaimer
from src.retriever import RetrievedChunk, retrieve_and_rerank
from src.utils import get_logger

logger = get_logger(__name__)


# ─── Prompt Templates ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are MedRAG, an advanced medical AI assistant designed for medical students, residents,
and healthcare professionals. You have access to excerpts from trusted medical literature.

Your goal is to give a clinically accurate, evidence-based answer. Follow these guidelines:
1. Base your answer primarily on the provided CONTEXT excerpts.
2. Cite the relevant excerpts (e.g., "[Excerpt 2]") when drawing from the context.
3. If the context is partial, supplement with established medical knowledge — clearly
   distinguishing context-grounded information from general medical knowledge.
4. Use proper medical terminology appropriate for healthcare professionals and students.
5. Structure your answer clearly: use numbered steps, bullet points, or headers as needed.
6. For pharmacology, include mechanisms of action, indications, and key contraindications.
7. For clinical reasoning, walk through differentials or diagnostic criteria where relevant.
8. Never recommend specific dosages for individual patients or replace clinical judgement.
"""


def _build_messages(context: str, question: str) -> list:
    """Build system + human message list for ChatGroq."""
    from langchain_core.messages import HumanMessage, SystemMessage
    user_text = (
        f"MEDICAL LITERATURE CONTEXT:\n"
        f"{context}\n\n"
        f"PATIENT QUESTION:\n{question}\n\n"
        f"Please provide a thorough, helpful answer based on the context above."
    )
    return [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_text)]


# Keep _PROMPT_TEMPLATE for backward-compat with any direct imports
from langchain_core.prompts import PromptTemplate as _PT
_PROMPT_TEMPLATE = _PT(
    input_variables=["context", "question"],
    template="{context}\n\nQuestion: {question}\nAnswer:",
)



# ─── LLM Singleton ────────────────────────────────────────────────────────────

_llm = None
_llm_model: str = ""   # tracks which model the singleton was built for


def _get_llm() -> ChatGroq:
    """Return a cached ChatGroq instance; rebuilds if GROQ_MODEL changed."""
    global _llm, _llm_model
    # Force a fresh read from .env (delete stale OS env vars first so override works)
    import os as _os
    from dotenv import load_dotenv
    from pathlib import Path as _Path
    _env_path = _Path(__file__).resolve().parent.parent / ".env"
    # Remove stale env vars so load_dotenv can overwrite them
    for _key in ("GROQ_MODEL", "GROQ_API_KEY"):
        _os.environ.pop(_key, None)
    load_dotenv(dotenv_path=_env_path, override=True)
    current_model = _os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    current_key   = _os.getenv("GROQ_API_KEY", "")

    if _llm is None or _llm_model != current_model:
        if not current_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Get a free API key at https://console.groq.com and add it to your .env file."
            )
        logger.info(f"Connecting to Groq API with model: {current_model}")
        _llm = ChatGroq(
            api_key=current_key,
            model=current_model,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        _llm_model = current_model
    return _llm



# ─── Context Formatting ───────────────────────────────────────────────────────

def format_context(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a structured context block for the prompt.

    Each chunk is labelled with its index and source metadata so the LLM
    can attribute its answer to specific documents.

    Args:
        chunks: List of RetrievedChunk objects.

    Returns:
        A formatted multi-line string to be inserted into the prompt.
    """
    if not chunks:
        return "No relevant context was found in the medical knowledge base."

    sections = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "Unknown source")
        page = chunk.metadata.get("page", "")
        page_str = f", page {page}" if page else ""
        sections.append(
            f"[Excerpt {i} — {source}{page_str}]\n{chunk.content.strip()}"
        )

    return "\n\n".join(sections)


# ─── Generation ───────────────────────────────────────────────────────────────

def generate(
    question: str,
    chunks: list[RetrievedChunk],
    apply_disclaimer: bool = True,
    check_grounded: bool = True,
) -> dict:
    """
    Generate a grounded medical response using Groq API.

    Args:
        question:         The user's question.
        chunks:           Retrieved context chunks from the retriever.
        apply_disclaimer: Whether to append the medical disclaimer.
        check_grounded:   Whether to run a grounding check on the response.

    Returns:
        A dict with:
          - 'answer'      (str):       The final response text.
          - 'sources'     (list):      Source metadata for each chunk used.
          - 'grounding'   (dict):      Grounding check results.
          - 'raw_answer'  (str):       The raw LLM output before disclaimer.
    """
    context = format_context(chunks)
    import os as _os
    logger.info(f"Generating response with Groq ({_os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')})...")

    llm = _get_llm()
    messages = _build_messages(context, question)

    try:
        result = llm.invoke(messages)
        raw_answer = result.content
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raw_answer = (
            "I encountered an error while generating a response. "
            "Please check that your GROQ_API_KEY and GROQ_MODEL are correctly set in the .env file."
        )

    # Run grounding check
    grounding_result = {"is_grounded": True, "warnings": []}
    if check_grounded:
        context_texts = [c.content for c in chunks]
        grounding_result = check_grounding(raw_answer, context_texts)
        if not grounding_result["is_grounded"]:
            logger.warning("Grounding check failed — response may not be fully supported by context.")

    # Inject disclaimer
    final_answer = inject_disclaimer(raw_answer) if apply_disclaimer else raw_answer

    # Collect source metadata
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

    logger.info("Response generation complete.")
    return {
        "answer": final_answer,
        "raw_answer": raw_answer,
        "sources": sources,
        "grounding": grounding_result,
    }


def generate_stream(question: str, chunks: list[RetrievedChunk]):
    """
    Stream response tokens in real-time as Groq generates them.
    Yields string chunks.
    """
    context = format_context(chunks)
    llm = _get_llm()
    messages = _build_messages(context, question)
    try:
        for token in llm.stream(messages):
            yield token.content
    except Exception as e:
        logger.error(f"Streaming generation failed: {e}")
        yield f"\n[Error generating response: {e}]"



def answer_question(question: str, top_k: int = 5) -> dict:
    """
    End-to-end convenience function: retrieve → rerank → generate.

    Args:
        question: The user's medical question.
        top_k:    Number of context chunks to retrieve.

    Returns:
        The full response dict from `generate()`.
    """
    chunks = retrieve_and_rerank(question, top_k=top_k)
    return generate(question, chunks)


# ─── CLI Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    question_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the risk factors for type 2 diabetes?"
    print(f"\nQuestion: {question_text}\n" + "-"*60)
    result = answer_question(question_text)
    print("\nANSWER:\n")
    sys.stdout.buffer.write(result["answer"].encode('utf-8'))
    print("\n" + "-"*60)
    print(f"SOURCES ({len(result['sources'])}):")
    for i, src in enumerate(result["sources"], 1):
        print(f"  [{i}] {src['source']} (chunk {src['chunk_index']}, score={src['score']})")
    if not result["grounding"]["is_grounded"]:
        print("\nGROUNDING WARNINGS:")
        for w in result["grounding"]["warnings"]:
            print(f"   - {w}")
