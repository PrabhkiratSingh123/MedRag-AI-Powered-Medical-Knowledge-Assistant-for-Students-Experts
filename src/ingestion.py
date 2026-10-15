"""
MedRAG Document Ingestion Pipeline
------------------------------------
This script loads medical documents from the configured directory,
splits them into overlapping chunks, generates local HuggingFace
embeddings, and upserts the results into MongoDB Atlas.

Supported formats: PDF, TXT, JSON/JSONL, CSV

Usage:
    python -m src.ingestion [--docs-dir PATH] [--reset]

Options:
    --docs-dir PATH   Override the documents directory from .env
    --reset           Drop the collection before ingesting (fresh start)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Generator

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
)
from langchain_core.documents import Document

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from tqdm import tqdm

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_DIR,
    EMBEDDING_MODEL,
    MONGODB_COLLECTION_NAME,
    MONGODB_DB_NAME,
    MONGODB_URI,
    get_mongo_client,
    validate_config,
)
from src.utils import clean_text, get_logger

logger = get_logger(__name__)


# ─── Document Loaders ────────────────────────────────────────────────────────

def load_pdf(path: Path) -> list[Document]:
    """Load a PDF file, returning one Document per page."""
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = path.name
        doc.metadata["file_type"] = "pdf"
    logger.info(f"Loaded PDF: {path.name} ({len(docs)} pages)")
    return docs


def load_txt(path: Path) -> list[Document]:
    """Load a plain text file as a single Document."""
    loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = path.name
        doc.metadata["file_type"] = "txt"
    logger.info(f"Loaded TXT: {path.name}")
    return docs


def load_json(path: Path) -> list[Document]:
    """
    Load a JSON or JSONL file.
    - JSONL: each line is a separate record → one Document each.
    - JSON array: each element is a record → one Document each.
    - JSON object: treated as a single Document.
    Looks for common text fields: 'text', 'content', 'body', 'abstract'.
    """
    docs: list[Document] = []
    text_keys = ["text", "content", "body", "abstract", "description"]

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            lines = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            lines = data if isinstance(data, list) else [data]

    for i, record in enumerate(lines):
        text = ""
        for key in text_keys:
            if key in record and isinstance(record[key], str):
                text = record[key]
                break
        if not text:
            # Fallback: join all string values
            text = " ".join(str(v) for v in record.values() if isinstance(v, str))
        if text.strip():
            metadata = {
                k: v
                for k, v in record.items()
                if isinstance(v, (str, int, float)) and k not in text_keys
            }
            metadata["source"] = path.name
            metadata["file_type"] = path.suffix.lstrip(".")
            metadata["record_index"] = i
            docs.append(Document(page_content=clean_text(text), metadata=metadata))

    logger.info(f"Loaded JSON/JSONL: {path.name} ({len(docs)} records)")
    return docs


def load_csv(path: Path) -> list[Document]:
    """
    Load a CSV file. Each row becomes a Document.
    Looks for common text columns: 'text', 'content', 'abstract', 'body'.
    If none found, concatenates all string columns.
    """
    df = pd.read_csv(path, dtype=str).fillna("")
    text_cols = [c for c in df.columns if c.lower() in {"text", "content", "abstract", "body", "description"}]
    docs: list[Document] = []

    for i, row in df.iterrows():
        if text_cols:
            text = " ".join(row[c] for c in text_cols)
        else:
            text = " | ".join(f"{col}: {val}" for col, val in row.items() if val)
        if text.strip():
            metadata = {col: row[col] for col in df.columns if col not in text_cols and row[col]}
            metadata["source"] = path.name
            metadata["file_type"] = "csv"
            metadata["row_index"] = i
            docs.append(Document(page_content=clean_text(text), metadata=metadata))

    logger.info(f"Loaded CSV: {path.name} ({len(docs)} rows)")
    return docs


LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".json": load_json,
    ".jsonl": load_json,
    ".csv": load_csv,
}


def discover_documents(docs_dir: str) -> Generator[Path, None, None]:
    """Yield all supported document paths in the given directory (recursive)."""
    base = Path(docs_dir)
    if not base.exists():
        logger.error(f"Documents directory does not exist: {base.resolve()}")
        sys.exit(1)
    for path in base.rglob("*"):
        if path.is_file() and path.suffix.lower() in LOADERS:
            yield path


# ─── Chunking ────────────────────────────────────────────────────────────────

def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into overlapping chunks using a recursive character splitter.
    This respects natural text boundaries (paragraphs, sentences).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    # Tag each chunk with its index within its source document
    source_counters: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        source_counters[src] = source_counters.get(src, 0) + 1
        chunk.metadata["chunk_index"] = source_counters[src]
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


# ─── Embedding & Upsert ──────────────────────────────────────────────────────

def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load the HuggingFace embedding model (downloaded once, cached locally)."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def upsert_to_mongodb(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
    reset: bool = False,
) -> None:
    """
    Store chunk embeddings in MongoDB Atlas.
    If reset=True, the collection is cleared before insertion.
    """
    client = get_mongo_client()
    db = client[MONGODB_DB_NAME]
    collection = db[MONGODB_COLLECTION_NAME]

    if reset:
        logger.warning("--reset flag set: dropping existing collection...")
        collection.drop()

    logger.info(f"Upserting {len(chunks)} chunks to MongoDB Atlas...")
    start = time.time()

    # MongoDBAtlasVectorSearch.from_documents handles batching and embedding
    MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection=collection,
        index_name="medrag_vector_index",  # Must match the Atlas Search index
    )

    elapsed = time.time() - start
    logger.info(f"✅ Upserted {len(chunks)} chunks in {elapsed:.1f}s")
    client.close()


def delete_uploaded_documents(filenames: list[str]) -> int:
    """
    Delete chunks matching the specified source filenames from MongoDB Atlas
    and remove the files from disk.

    Args:
        filenames: List of file names (e.g. ['report.pdf', 'data.txt']).

    Returns:
        Number of chunks deleted from MongoDB Atlas.
    """
    if not filenames:
        return 0

    client = get_mongo_client()
    db = client[MONGODB_DB_NAME]
    collection = db[MONGODB_COLLECTION_NAME]

    res = collection.delete_many({"source": {"$in": filenames}})
    deleted_count = res.deleted_count
    logger.info(f"Deleted {deleted_count} chunk(s) from MongoDB Atlas for files: {filenames}")

    docs_dir = Path(DOCS_DIR)
    for name in filenames:
        fp = docs_dir / name
        if fp.exists():
            try:
                fp.unlink()
                logger.info(f"Deleted file from disk: {name}")
            except Exception as e:
                logger.warning(f"Could not delete file {name}: {e}")

    client.close()
    return deleted_count



# ─── Main ────────────────────────────────────────────────────────────────────

def run_ingestion(docs_dir: str = DOCS_DIR, reset: bool = False) -> None:
    """Full ingestion pipeline: load → chunk → embed → store."""
    warnings = validate_config()
    if warnings:
        for w in warnings:
            logger.warning(w)

    # 1. Discover and load all documents
    logger.info(f"Scanning documents in: {Path(docs_dir).resolve()}")
    all_docs: list[Document] = []
    doc_paths = list(discover_documents(docs_dir))

    if not doc_paths:
        logger.error("No supported documents found. Exiting.")
        sys.exit(1)

    for path in tqdm(doc_paths, desc="Loading documents"):
        loader_fn = LOADERS[path.suffix.lower()]
        try:
            all_docs.extend(loader_fn(path))
        except Exception as e:
            logger.error(f"Failed to load {path.name}: {e}")

    logger.info(f"Total documents loaded: {len(all_docs)}")

    # 2. Chunk documents
    chunks = chunk_documents(all_docs)

    # 3. Load embedding model
    embeddings = get_embedding_model()

    # 4. Upsert to MongoDB
    upsert_to_mongodb(chunks, embeddings, reset=reset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedRAG Document Ingestion Pipeline")
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=DOCS_DIR,
        help="Path to the directory containing medical documents",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the MongoDB collection before ingesting (fresh start)",
    )
    args = parser.parse_args()
    run_ingestion(docs_dir=args.docs_dir, reset=args.reset)
