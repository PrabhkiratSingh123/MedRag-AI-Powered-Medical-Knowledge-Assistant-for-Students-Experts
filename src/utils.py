"""
MedRAG Utilities
----------------
Shared helper functions used across multiple modules.
"""

import logging
import sys
from typing import Any
import colorlog


# ─── Logging Setup ──────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Returns a colour-formatted logger for the given module name.

    Usage:
        logger = get_logger(__name__)
        logger.info("Starting ingestion...")
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Avoid adding duplicate handlers if already configured
        return logger

    logger.setLevel(logging.DEBUG)

    handler = colorlog.StreamHandler(sys.stdout)
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s | %(cyan)s%(name)s%(reset)s | %(message)s",
            log_colors={
                "DEBUG": "white",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ─── Text Helpers ────────────────────────────────────────────────────────────

def truncate_text(text: str, max_chars: int = 300) -> str:
    """Truncate text for display, appending '...' if shortened."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def format_source_metadata(metadata: dict[str, Any]) -> str:
    """
    Format chunk metadata into a human-readable source citation string.

    Args:
        metadata: A dict with keys like 'source', 'page', 'chunk_index'.

    Returns:
        A formatted citation string.
    """
    parts = []
    if source := metadata.get("source"):
        parts.append(f"📄 **Source**: {source}")
    if page := metadata.get("page"):
        parts.append(f"📃 **Page**: {page}")
    if chunk_index := metadata.get("chunk_index") is not None:
        parts.append(f"🔢 **Chunk**: {metadata.get('chunk_index')}")
    return " | ".join(parts) if parts else "Source: Unknown"


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Collapse excessive whitespace / newlines
    - Strip leading/trailing whitespace
    """
    import re
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
