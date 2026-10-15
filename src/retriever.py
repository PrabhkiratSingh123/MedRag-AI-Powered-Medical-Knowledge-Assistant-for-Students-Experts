"""
MedRAG Retriever
-----------------
Handles query embedding and semantic retrieval from MongoDB Atlas Vector Search.

Steps:
  1. Embed the user query using the same HuggingFace model used during ingestion.
  2. Run a vector similarity search against the MongoDB Atlas collection.
  3. Optionally re-rank results by a cross-encoder for improved precision.

Usage (standalone test):
    python -m src.retriever "What are the symptoms of type 2 diabetes?"
"""

import sys
from dataclasses import dataclass, field
from typing import Optional

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

from src.config import (
    EMBEDDING_MODEL,
    MONGODB_COLLECTION_NAME,
    MONGODB_DB_NAME,
    MONGODB_URI,
    MONGODB_VECTOR_INDEX,
    TOP_K,
    get_mongo_client,
    validate_config,
)
from src.utils import format_source_metadata, get_logger, truncate_text

logger = get_logger(__name__)


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with its metadata and relevance score."""
    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0

    @property
    def source_citation(self) -> str:
        """Human-readable source citation string."""
        return format_source_metadata(self.metadata)

    @property
    def preview(self) -> str:
        """Short preview of the chunk content."""
        return truncate_text(self.content, max_chars=200)


# ─── Singleton Embedding Model ────────────────────────────────────────────────
# Load once per session to avoid re-downloading / re-initialising on every call.
_embedding_model: Optional[HuggingFaceEmbeddings] = None


def _get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


# ─── MongoDB Vector Store ─────────────────────────────────────────────────────

def _get_vector_store() -> MongoDBAtlasVectorSearch:
    """Return a LangChain MongoDBAtlasVectorSearch instance."""
    client = get_mongo_client()
    collection = client[MONGODB_DB_NAME][MONGODB_COLLECTION_NAME]
    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=_get_embedding_model(),
        index_name=MONGODB_VECTOR_INDEX,
        text_key="text",
        embedding_key="embedding",
    )


# ─── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
    """
    Retrieve the top-K most semantically relevant chunks for a user query.

    Args:
        query:  The user's question.
        top_k:  Maximum number of chunks to return.

    Returns:
        A list of RetrievedChunk objects, ordered by descending similarity.
    """
    warnings = validate_config()
    if warnings:
        for w in warnings:
            logger.warning(w)

    logger.info(f"Retrieving top-{top_k} chunks for query: '{query[:80]}'")

    try:
        vector_store = _get_vector_store()
        results = vector_store.similarity_search_with_score(query, k=top_k)
    except Exception as e:
        logger.error(f"MongoDB vector search failed: {e}")
        return []

    chunks = [
        RetrievedChunk(
            content=doc.page_content,
            metadata=doc.metadata,
            score=float(score),
        )
        for doc, score in results
    ]

    logger.info(
        f"Retrieved {len(chunks)} chunks. "
        f"Top score: {chunks[0].score:.4f}" if chunks else "No chunks found."
    )
    return chunks


def rerank(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Re-rank retrieved chunks using a cross-encoder model for improved relevance.

    Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` which is lightweight and runs locally.
    Falls back to the original order if the cross-encoder is unavailable.

    Args:
        query:  The original user query.
        chunks: The list of chunks from the initial vector search.

    Returns:
        Re-ranked list of RetrievedChunk objects.
    """
    if not chunks:
        return chunks

    try:
        from sentence_transformers import CrossEncoder

        logger.info("Reranking results with cross-encoder...")
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, chunk.content) for chunk in chunks]
        scores = cross_encoder.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        reranked = sorted(chunks, key=lambda c: c.score, reverse=True)
        logger.info("Reranking complete.")
        return reranked

    except ImportError:
        logger.warning(
            "sentence-transformers CrossEncoder not available. "
            "Skipping reranking — results returned in vector search order."
        )
        return chunks
    except Exception as e:
        logger.warning(f"Reranking failed ({e}). Returning original order.")
        return chunks


def retrieve_and_rerank(query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
    """
    Convenience function: retrieve then rerank in one call.

    Args:
        query:  The user's question.
        top_k:  Number of chunks to retrieve before reranking.

    Returns:
        Re-ranked list of RetrievedChunk objects.
    """
    chunks = retrieve(query, top_k=top_k)
    return rerank(query, chunks)


# ─── CLI Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What causes type 2 diabetes?"
    print(f"\n🔍 Query: {query_text}\n{'─'*60}")
    results = retrieve_and_rerank(query_text)
    if not results:
        print("No results found.")
    for i, chunk in enumerate(results, 1):
        print(f"\n[{i}] Score: {chunk.score:.4f}")
        print(f"    {chunk.source_citation}")
        print(f"    {chunk.preview}")
