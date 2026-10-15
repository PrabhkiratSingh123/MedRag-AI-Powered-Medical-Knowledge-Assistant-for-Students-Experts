"""
MedRAG Database Layer
----------------------
Full CRUD operations and MongoDB Aggregation Pipelines
for the MedRAG medical knowledge base.

Collections:
  • medical_chunks  — vector-embedded document chunks (main RAG store)
  • query_logs      — user query history for analytics

CRUD:
  create_chunk        Insert a single knowledge chunk manually
  read_chunks         Paginated read with optional source/keyword filter
  update_chunk_meta   Update metadata fields on a chunk by _id
  update_chunk_text   Update text content of a chunk (marks embedding stale)
  delete_chunk        Delete a single chunk by _id
  delete_by_source    Delete all chunks belonging to a source file

Aggregation Pipelines (all support optional `sources` list filter):
  agg_knowledge_base_summary   High-level stats (total, sources, avg size)
  agg_chunks_per_source        Count + avg length per source document
  agg_filetype_distribution    File type breakdown with percentages
  agg_chunk_length_buckets     Length distribution via $bucket stage
  agg_stale_embeddings         Count of stale vs healthy embeddings
  agg_top_sources              Top-N largest sources by chunk count
  log_query                    Write a query event to query_logs
  agg_query_history            Recent query logs with timestamps
  agg_query_stats              Performance summary over query_logs
"""

from __future__ import annotations

import datetime
from typing import Any, Optional

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from src.config import (
    MONGODB_DB_NAME,
    MONGODB_COLLECTION_NAME,
    get_mongo_client,
)
from src.utils import get_logger

logger = get_logger(__name__)

QUERY_LOGS_COLLECTION = "query_logs"


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_collection(name: str = MONGODB_COLLECTION_NAME):
    """Return (client, collection). Caller must call client.close()."""
    client = get_mongo_client()
    return client, client[MONGODB_DB_NAME][name]


def _serialize(doc: dict) -> dict:
    """Convert ObjectId / non-serialisable types for display."""
    out = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            out[k] = str(v)
        elif isinstance(v, list) and k == "embedding":
            out[k] = f"[vector dim={len(v)}]"
        elif isinstance(v, (datetime.datetime, datetime.date)):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


def _source_match(sources: Optional[list[str]]) -> dict:
    """Build a MongoDB $match stage dict that filters by source list (or empty for all)."""
    if sources:
        return {"source": {"$in": sources}}
    return {}


# ══════════════════════════════════════════════════════════════════════════════
#  CREATE
# ══════════════════════════════════════════════════════════════════════════════

def create_chunk(
    text: str,
    source: str,
    file_type: str = "manual",
    page: Optional[int] = None,
    extra_meta: Optional[dict] = None,
) -> str:
    """
    Manually insert a knowledge chunk into the medical_chunks collection.

    Note: This inserts WITHOUT an embedding vector, so it will not appear
    in vector-similarity search results. Use the ingestion pipeline (sidebar
    upload) for fully-embedded documents.

    Args:
        text:       The chunk text content.
        source:     Source identifier (e.g., 'manual_entry' or a filename).
        file_type:  Type label (default 'manual').
        page:       Optional page number.
        extra_meta: Any additional metadata fields to store.

    Returns:
        The inserted document's _id as a string.
    """
    doc: dict[str, Any] = {
        "text":        text,
        "source":      source,
        "file_type":   file_type,
        "chunk_index": 0,
        "created_at":  datetime.datetime.utcnow(),
        "embedding":   [],   # empty — not searchable via vector index
    }
    if page is not None:
        doc["page"] = page
    if extra_meta:
        doc.update(extra_meta)

    client, col = _get_collection()
    try:
        result     = col.insert_one(doc)
        inserted_id = str(result.inserted_id)
        logger.info(f"[CREATE] Inserted chunk id={inserted_id} source='{source}'")
        return inserted_id
    finally:
        client.close()


# ══════════════════════════════════════════════════════════════════════════════
#  READ
# ══════════════════════════════════════════════════════════════════════════════

def read_chunks(
    source_filter: Optional[str] = None,
    keyword: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
    sources_scope: Optional[list[str]] = None,
) -> tuple[list[dict], int]:
    """
    Paginated read of chunks with optional filters.

    Args:
        source_filter:  Exact source filename to filter by (or None for all).
        keyword:        Case-insensitive substring to match in text (or None).
        skip:           Number of documents to skip (pagination).
        limit:          Maximum documents to return.
        sources_scope:  If set, only search within these source filenames
                        (used to scope results to the current session's uploads).

    Returns:
        (list_of_serialised_docs, total_count_matching_filter)
    """
    query: dict[str, Any] = {}

    # Scope to session uploads if provided
    if sources_scope:
        query["source"] = {"$in": sources_scope}

    # Override / narrow by specific source
    if source_filter and source_filter != "(All)":
        query["source"] = source_filter

    if keyword:
        query["text"] = {"$regex": keyword, "$options": "i"}

    client, col = _get_collection()
    try:
        total  = col.count_documents(query)
        cursor = (
            col.find(query, {"embedding": 0})
               .sort("_id", DESCENDING)
               .skip(skip)
               .limit(limit)
        )
        docs = [_serialize(d) for d in cursor]
        logger.info(f"[READ] Fetched {len(docs)}/{total} chunks (filter={query})")
        return docs, total
    finally:
        client.close()


def get_all_sources(sources_scope: Optional[list[str]] = None) -> list[str]:
    """
    Return a sorted list of all unique source filenames.

    Args:
        sources_scope: If set, only return sources within this list.
    """
    client, col = _get_collection()
    try:
        if sources_scope:
            sources = col.distinct("source", {"source": {"$in": sources_scope}})
        else:
            sources = col.distinct("source")
        return sorted(s for s in sources if s)
    finally:
        client.close()


def get_chunk_by_id(chunk_id: str) -> Optional[dict]:
    """Fetch a single chunk by its string _id. Returns None if not found."""
    client, col = _get_collection()
    try:
        doc = col.find_one({"_id": ObjectId(chunk_id)}, {"embedding": 0})
        return _serialize(doc) if doc else None
    except Exception as e:
        logger.warning(f"[READ] get_chunk_by_id failed: {e}")
        return None
    finally:
        client.close()


# ══════════════════════════════════════════════════════════════════════════════
#  UPDATE
# ══════════════════════════════════════════════════════════════════════════════

def update_chunk_meta(chunk_id: str, updates: dict[str, Any]) -> bool:
    """
    Update metadata fields on a chunk document.
    Protects core fields (text, embedding) from accidental overwrite.

    Args:
        chunk_id: The string _id of the chunk.
        updates:  Dict of field-name → new-value pairs to set.

    Returns:
        True if the document was found and modified, False otherwise.
    """
    protected   = {"_id", "embedding", "text"}
    safe_updates = {k: v for k, v in updates.items() if k not in protected}
    if not safe_updates:
        logger.warning("[UPDATE] No safe fields to update.")
        return False

    safe_updates["updated_at"] = datetime.datetime.utcnow()

    client, col = _get_collection()
    try:
        result   = col.update_one(
            {"_id": ObjectId(chunk_id)},
            {"$set": safe_updates},
        )
        modified = result.modified_count > 0
        logger.info(f"[UPDATE] id={chunk_id} modified={modified} fields={list(safe_updates.keys())}")
        return modified
    except Exception as e:
        logger.error(f"[UPDATE] Failed: {e}")
        return False
    finally:
        client.close()


def update_chunk_text(chunk_id: str, new_text: str) -> bool:
    """
    Update the text content of a chunk.
    Marks embedding_stale=True since the vector will no longer match the text.

    Args:
        chunk_id: The string _id of the chunk.
        new_text: The replacement text.

    Returns:
        True if modified, False otherwise.
    """
    client, col = _get_collection()
    try:
        result   = col.update_one(
            {"_id": ObjectId(chunk_id)},
            {"$set": {
                "text":            new_text,
                "updated_at":      datetime.datetime.utcnow(),
                "embedding_stale": True,
            }},
        )
        modified = result.modified_count > 0
        logger.info(f"[UPDATE TEXT] id={chunk_id} modified={modified}")
        return modified
    except Exception as e:
        logger.error(f"[UPDATE TEXT] Failed: {e}")
        return False
    finally:
        client.close()


# ══════════════════════════════════════════════════════════════════════════════
#  DELETE
# ══════════════════════════════════════════════════════════════════════════════

def delete_chunk(chunk_id: str) -> bool:
    """
    Delete a single chunk by its _id.

    Returns:
        True if a document was deleted, False if not found.
    """
    client, col = _get_collection()
    try:
        result  = col.delete_one({"_id": ObjectId(chunk_id)})
        deleted = result.deleted_count > 0
        logger.info(f"[DELETE] id={chunk_id} deleted={deleted}")
        return deleted
    except Exception as e:
        logger.error(f"[DELETE] Failed: {e}")
        return False
    finally:
        client.close()


def delete_by_source(source_name: str) -> int:
    """
    Delete all chunks belonging to a specific source document.

    Args:
        source_name: The exact source filename (e.g., 'cardiology.pdf').

    Returns:
        Number of chunks deleted.
    """
    client, col = _get_collection()
    try:
        result = col.delete_many({"source": source_name})
        count  = result.deleted_count
        logger.info(f"[DELETE SOURCE] source='{source_name}' deleted={count} chunks")
        return count
    finally:
        client.close()


def delete_session_files(filenames: list[str]) -> int:
    """
    Delete all chunks for a list of source filenames (session cleanup).
    Also removes the physical files from the documents directory.

    Args:
        filenames: List of filenames that were uploaded this session.

    Returns:
        Total chunks deleted across all files.
    """
    if not filenames:
        return 0

    from src.config import DOCS_DIR
    from pathlib import Path

    client, col = _get_collection()
    total = 0
    try:
        result = col.delete_many({"source": {"$in": filenames}})
        total  = result.deleted_count
        logger.info(f"[DELETE SESSION] Deleted {total} chunks for files: {filenames}")
    finally:
        client.close()

    # Remove files from disk
    docs_dir = Path(DOCS_DIR)
    for name in filenames:
        fp = docs_dir / name
        if fp.exists():
            try:
                fp.unlink()
                logger.info(f"[DELETE SESSION] Removed file from disk: {name}")
            except Exception as e:
                logger.warning(f"[DELETE SESSION] Could not delete {name}: {e}")

    return total


def delete_all_chunks() -> int:
    """
    ⚠️ Delete ALL chunks from the collection (full reset).

    Returns:
        Number of chunks deleted.
    """
    client, col = _get_collection()
    try:
        result = col.delete_many({})
        count  = result.deleted_count
        logger.warning(f"[DELETE ALL] Cleared {count} chunks from collection.")
        return count
    finally:
        client.close()


# ══════════════════════════════════════════════════════════════════════════════
#  AGGREGATION PIPELINES
#  All accept an optional `sources` list to scope to session-uploaded files.
# ══════════════════════════════════════════════════════════════════════════════

def agg_knowledge_base_summary(sources: Optional[list[str]] = None) -> dict:
    """
    Aggregation Pipeline 1 — Knowledge Base Summary.

    Args:
        sources: If provided, compute stats only for these source filenames.

    Returns dict with: total_chunks, unique_sources, avg_text_length, total_characters
    """
    match = _source_match(sources)
    pipeline = [
        *([ {"$match": match} ] if match else []),
        {
            "$group": {
                "_id":            None,
                "total_chunks":   {"$sum": 1},
                "unique_sources": {"$addToSet": "$source"},
                "avg_text_length":  {"$avg": {"$strLenCP": "$text"}},
                "total_characters": {"$sum": {"$strLenCP": "$text"}},
            }
        },
        {
            "$project": {
                "_id": 0,
                "total_chunks":     1,
                "unique_sources":   {"$size": "$unique_sources"},
                "avg_text_length":  {"$round": ["$avg_text_length", 0]},
                "total_characters": 1,
            }
        },
    ]
    client, col = _get_collection()
    try:
        result = list(col.aggregate(pipeline))
        return result[0] if result else {
            "total_chunks": 0, "unique_sources": 0,
            "avg_text_length": 0, "total_characters": 0,
        }
    finally:
        client.close()


def agg_chunks_per_source(sources: Optional[list[str]] = None) -> list[dict]:
    """
    Aggregation Pipeline 2 — Chunks per Source.

    Groups by source filename, counts chunks, computes avg length.

    Args:
        sources: If provided, only include these source filenames.

    Returns:
        [{"source": str, "chunk_count": int, "avg_length": float, "file_type": str}, ...]
    """
    match = _source_match(sources)
    pipeline = [
        *([ {"$match": match} ] if match else []),
        {
            "$group": {
                "_id":         "$source",
                "chunk_count": {"$sum": 1},
                "avg_length":  {"$avg": {"$strLenCP": "$text"}},
                "file_type":   {"$first": "$file_type"},
            }
        },
        {"$sort": {"chunk_count": DESCENDING}},
        {
            "$project": {
                "_id": 0,
                "source":      "$_id",
                "chunk_count": 1,
                "avg_length":  {"$round": ["$avg_length", 0]},
                "file_type":   1,
            }
        },
    ]
    client, col = _get_collection()
    try:
        return list(col.aggregate(pipeline))
    finally:
        client.close()


def agg_filetype_distribution(sources: Optional[list[str]] = None) -> list[dict]:
    """
    Aggregation Pipeline 3 — File Type Distribution.

    Args:
        sources: If provided, compute distribution only for these sources.

    Returns:
        [{"file_type": str, "count": int, "pct": float}, ...]
    """
    match = _source_match(sources)
    pipeline = [
        *([ {"$match": match} ] if match else []),
        {"$group": {"_id": "$file_type", "count": {"$sum": 1}}},
        {
            "$group": {
                "_id":   None,
                "types": {"$push": {"file_type": "$_id", "count": "$count"}},
                "total": {"$sum": "$count"},
            }
        },
        {"$unwind": "$types"},
        {
            "$project": {
                "_id":       0,
                "file_type": "$types.file_type",
                "count":     "$types.count",
                "pct": {
                    "$round": [
                        {"$multiply": [{"$divide": ["$types.count", "$total"]}, 100]},
                        1,
                    ]
                },
            }
        },
        {"$sort": {"count": DESCENDING}},
    ]
    client, col = _get_collection()
    try:
        return list(col.aggregate(pipeline))
    finally:
        client.close()


def agg_chunk_length_buckets(sources: Optional[list[str]] = None) -> list[dict]:
    """
    Aggregation Pipeline 4 — Chunk Length Distribution (bucketed via $bucket).

    Buckets: Short(<200), Medium(200-600), Standard(600-1000), Long(>1000)

    Args:
        sources: If provided, compute buckets only for these sources.

    Returns:
        [{"bucket": str, "count": int}, ...]
    """
    match = _source_match(sources)
    pipeline = [
        *([ {"$match": match} ] if match else []),
        {
            "$bucket": {
                "groupBy":    {"$strLenCP": "$text"},
                "boundaries": [0, 200, 600, 1000, 10000],
                "default":    "other",
                "output":     {"count": {"$sum": 1}},
            }
        },
        {
            "$project": {
                "_id": 0,
                "bucket": {
                    "$switch": {
                        "branches": [
                            {"case": {"$eq": ["$_id", 0]},    "then": "Short (<200 chars)"},
                            {"case": {"$eq": ["$_id", 200]},  "then": "Medium (200–600)"},
                            {"case": {"$eq": ["$_id", 600]},  "then": "Standard (600–1000)"},
                            {"case": {"$eq": ["$_id", 1000]}, "then": "Long (>1000 chars)"},
                        ],
                        "default": "Other",
                    }
                },
                "count": 1,
            }
        },
    ]
    client, col = _get_collection()
    try:
        return list(col.aggregate(pipeline))
    finally:
        client.close()


def agg_stale_embeddings(sources: Optional[list[str]] = None) -> dict:
    """
    Aggregation Pipeline 5 — Stale Embedding Detection.

    Args:
        sources: If provided, check only these sources.

    Returns:
        {"stale_count": int, "ok_count": int}
    """
    match = _source_match(sources)
    pipeline = [
        *([ {"$match": match} ] if match else []),
        {
            "$group": {
                "_id": None,
                "stale_count": {
                    "$sum": {"$cond": [{"$eq": ["$embedding_stale", True]}, 1, 0]}
                },
                "ok_count": {
                    "$sum": {"$cond": [{"$ne": ["$embedding_stale", True]}, 1, 0]}
                },
            }
        },
        {"$project": {"_id": 0, "stale_count": 1, "ok_count": 1}},
    ]
    client, col = _get_collection()
    try:
        result = list(col.aggregate(pipeline))
        return result[0] if result else {"stale_count": 0, "ok_count": 0}
    finally:
        client.close()


def agg_top_sources(n: int = 10, sources: Optional[list[str]] = None) -> list[dict]:
    """
    Aggregation Pipeline 6 — Top-N Sources by chunk count.

    Args:
        n:       How many top sources to return.
        sources: If provided, rank only among these sources.

    Returns:
        [{"source": str, "chunk_count": int}, ...]
    """
    match = _source_match(sources)
    pipeline = [
        *([ {"$match": match} ] if match else []),
        {"$group": {"_id": "$source", "chunk_count": {"$sum": 1}}},
        {"$sort": {"chunk_count": DESCENDING}},
        {"$limit": n},
        {"$project": {"_id": 0, "source": "$_id", "chunk_count": 1}},
    ]
    client, col = _get_collection()
    try:
        return list(col.aggregate(pipeline))
    finally:
        client.close()


# ══════════════════════════════════════════════════════════════════════════════
#  QUERY LOG
# ══════════════════════════════════════════════════════════════════════════════

def log_query(query: str, chunks_retrieved: int, elapsed_s: float,
              sources_used: Optional[list[str]] = None) -> None:
    """
    Append a query event to the query_logs collection.

    Args:
        query:            The user's raw query string.
        chunks_retrieved: How many chunks were retrieved.
        elapsed_s:        Total pipeline time in seconds.
        sources_used:     List of source filenames that contributed context.
    """
    doc: dict[str, Any] = {
        "query":            query,
        "chunks_retrieved": chunks_retrieved,
        "elapsed_s":        round(elapsed_s, 3),
        "timestamp":        datetime.datetime.utcnow(),
    }
    if sources_used:
        doc["sources_used"] = sources_used

    client, col = _get_collection(QUERY_LOGS_COLLECTION)
    try:
        col.insert_one(doc)
    except Exception as e:
        logger.warning(f"[LOG QUERY] Failed: {e}")
    finally:
        client.close()


def agg_query_history(limit: int = 30) -> list[dict]:
    """
    Return the most recent queries with timestamp and performance stats.

    Returns:
        [{"query": str, "chunks_retrieved": int, "elapsed_s": float,
          "sources_used": list, "timestamp": str}, ...]
    """
    client, col = _get_collection(QUERY_LOGS_COLLECTION)
    try:
        docs = list(
            col.find({}, {"_id": 0})
               .sort("timestamp", DESCENDING)
               .limit(limit)
        )
        for d in docs:
            if isinstance(d.get("timestamp"), datetime.datetime):
                d["timestamp"] = d["timestamp"].strftime("%Y-%m-%d %H:%M UTC")
        return docs
    finally:
        client.close()


def agg_query_stats() -> dict:
    """
    Aggregation Pipeline 7 — Query Performance Summary over query_logs.

    Returns:
        {"total_queries": int, "avg_elapsed_s": float,
         "max_elapsed_s": float, "avg_chunks": float}
    """
    pipeline = [
        {
            "$group": {
                "_id":           None,
                "total_queries": {"$sum": 1},
                "avg_elapsed_s": {"$avg": "$elapsed_s"},
                "max_elapsed_s": {"$max": "$elapsed_s"},
                "avg_chunks":    {"$avg": "$chunks_retrieved"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "total_queries": 1,
                "avg_elapsed_s": {"$round": ["$avg_elapsed_s", 2]},
                "max_elapsed_s": {"$round": ["$max_elapsed_s", 2]},
                "avg_chunks":    {"$round": ["$avg_chunks",    1]},
            }
        },
    ]
    client, col = _get_collection(QUERY_LOGS_COLLECTION)
    try:
        result = list(col.aggregate(pipeline))
        return result[0] if result else {
            "total_queries": 0, "avg_elapsed_s": 0.0,
            "max_elapsed_s": 0.0, "avg_chunks": 0.0,
        }
    finally:
        client.close()
