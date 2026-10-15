"""
MedRAG Safety Guardrails
-------------------------
Implements medical-specific safety checks applied to every user query
and every generated response.

Components:
  1. Emergency Query Detection   — detects acute emergency signals
  2. Out-of-Domain Detection     — detects clearly non-medical questions
  3. Disclaimer Injection        — appends a medical disclaimer to all answers
  4. Grounding Check             — verifies the answer is supported by context
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from src.config import MEDICAL_DISCLAIMER
from src.utils import get_logger

logger = get_logger(__name__)


# ─── Emergency Detection ────────────────────────────────────────────────────

# Phrases that strongly suggest the user or someone nearby is in acute danger.
# Kept intentionally broad to avoid false negatives — safety over precision.
_EMERGENCY_PATTERNS: list[str] = [
    r"\bhaving\s+a?\s*(heart\s+attack|stroke|seizure|cardiac\s+arrest)\b",
    r"\b(chest\s+pain|chest\s+tightness)\b.{0,30}\b(severe|crushing|can'?t\s+breathe)\b",
    r"\bcan'?t\s+breathe\b",
    r"\b(unconscious|not\s+breathing|not\s+responding|collapsed)\b",
    r"\bsuicid(e|al)\b",
    r"\bself.?harm\b",
    r"\boverdos(e|ed|ing)\b",
    r"\bpoisoned?\b",
    r"\bsevere\s+(bleeding|allergic\s+reaction|anaphylaxis)\b",
    r"\banaphylaxis\b",
    r"\bstroke\s+symptoms?\b",
    r"\bcall\s+(911|999|112|ambulance)\b",
    r"\bemergency\s+room\b.{0,20}\bnow\b",
    r"\bdying\b.{0,20}\bright\s+now\b",
]

_EMERGENCY_REGEX = re.compile(
    "|".join(_EMERGENCY_PATTERNS), re.IGNORECASE | re.DOTALL
)

EMERGENCY_RESPONSE = (
    "🚨 **This appears to be a medical emergency.** 🚨\n\n"
    "**Please call emergency services immediately (911, 999, or your local "
    "emergency number).**\n\n"
    "Do not wait for an AI response — go to your nearest emergency room or "
    "call for an ambulance right now.\n\n"
    "If someone is unconscious or not breathing:\n"
    "- Call emergency services\n"
    "- Begin CPR if you are trained\n"
    "- Do not leave the person alone\n\n"
    "---\n"
    "*MedRAG is an informational tool and cannot assist with medical emergencies.*"
)


# ─── Out-of-Domain Detection ─────────────────────────────────────────────────

# Patterns that signal clearly non-medical queries.
_OFF_TOPIC_PATTERNS: list[str] = [
    r"\b(write\s+(me\s+)?(a\s+)?(poem|song|story|essay|code|script))\b",
    r"\b(stock\s+price|cryptocurrency|bitcoin|forex)\b",
    r"\b(weather\s+(in|forecast))\b",
    r"\b(latest\s+(news|sports|scores))\b",
    r"\b(recipe\s+for|how\s+to\s+cook)\b",
    r"\b(who\s+(won|is\s+winning))\b.{0,20}\b(game|match|election)\b",
]

_OFF_TOPIC_REGEX = re.compile(
    "|".join(_OFF_TOPIC_PATTERNS), re.IGNORECASE | re.DOTALL
)

OFF_TOPIC_RESPONSE = (
    "I'm MedRAG, a medical AI assistant for healthcare students and professionals. "
    "I'm focused exclusively on medical topics.\n\n"
    "Your question appears to be outside my domain. Please ask me about clinical "
    "conditions, pathophysiology, pharmacology, diagnostic criteria, treatment "
    "protocols, or evidence-based medicine."
)


# ─── Grounding Check ─────────────────────────────────────────────────────────

# Keywords that suggest the LLM is hallucinating (generating without context)
_HALLUCINATION_SIGNALS: list[str] = [
    "as a language model",
    "i don't have access to",
    "i cannot provide",
    "based on my training",
    "my knowledge cutoff",
    "i'm not sure, but",
    "i believe, but i'm not certain",
]


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    """Result of a guardrail check on a user query."""
    is_emergency: bool = False
    is_off_topic: bool = False
    blocked: bool = False
    override_response: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


# ─── Public API ──────────────────────────────────────────────────────────────

def check_query(query: str) -> GuardrailResult:
    """
    Run all pre-generation guardrail checks on a user query.

    Returns a GuardrailResult. If `blocked=True`, use `override_response`
    instead of running the RAG pipeline.

    Args:
        query: The raw user query string.

    Returns:
        GuardrailResult with flags and optional override response.
    """
    result = GuardrailResult()

    # 1. Emergency detection (highest priority)
    if _EMERGENCY_REGEX.search(query):
        logger.warning(f"🚨 Emergency query detected: '{query[:80]}...'")
        result.is_emergency = True
        result.blocked = True
        result.override_response = EMERGENCY_RESPONSE
        return result  # Short-circuit immediately

    # 2. Off-topic detection
    if _OFF_TOPIC_REGEX.search(query):
        logger.info(f"Off-topic query detected: '{query[:80]}'")
        result.is_off_topic = True
        result.blocked = True
        result.override_response = OFF_TOPIC_RESPONSE
        return result

    return result


def inject_disclaimer(response: str) -> str:
    """
    Append the standard medical disclaimer to a generated response.

    Args:
        response: The raw LLM-generated response text.

    Returns:
        Response with disclaimer appended.
    """
    return response + MEDICAL_DISCLAIMER


def check_grounding(response: str, context_chunks: list[str]) -> dict:
    """
    Lightweight grounding check to flag potential hallucinations.

    Checks:
    1. Whether the response contains known hallucination signal phrases.
    2. Whether the response has *any* lexical overlap with the context.

    Args:
        response:       The LLM-generated response.
        context_chunks: The retrieved context chunks used to generate the response.

    Returns:
        A dict with:
          - 'is_grounded' (bool): True if the response appears grounded.
          - 'warnings' (list[str]): Any grounding warnings detected.
    """
    warnings = []

    # Check for hallucination signal phrases
    response_lower = response.lower()
    for signal in _HALLUCINATION_SIGNALS:
        if signal in response_lower:
            warnings.append(
                f"Possible hallucination signal detected: '{signal}'"
            )

    # Check lexical overlap with context
    if context_chunks:
        # Build a simple word set from all context chunks
        context_words = set(
            re.findall(r"\b[a-zA-Z]{4,}\b", " ".join(context_chunks).lower())
        )
        response_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", response_lower))
        overlap = response_words & context_words

        if len(context_words) > 0:
            overlap_ratio = len(overlap) / max(len(response_words), 1)
            if overlap_ratio < 0.05 and len(context_words) > 10:
                warnings.append(
                    f"Low lexical overlap ({overlap_ratio:.0%}) between response "
                    "and retrieved context. Response may not be grounded."
                )

    is_grounded = len(warnings) == 0

    if warnings:
        for w in warnings:
            logger.warning(f"Grounding check: {w}")

    return {"is_grounded": is_grounded, "warnings": warnings}
