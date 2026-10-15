"""
MedRAG Agent State
------------------
Defines the shared mutable state (TypedDict) threaded through every
LangGraph node in the MedRAG agentic graph.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Shared state object passed between all LangGraph nodes."""

    # ── Conversation ────────────────────────────────────────────────────────
    # Raw user message for the current turn
    user_query: str
    # Full conversation history as plain dicts {role, content}
    chat_history: list[dict[str, str]]

    # ── Routing ─────────────────────────────────────────────────────────────
    # "clinical" | "booking" | "emergency" | "off_topic"
    intent: str

    # ── RAG output ──────────────────────────────────────────────────────────
    rag_answer: str
    rag_sources: list[dict[str, Any]]
    rag_grounding: dict[str, Any]
    # True when the LLM classifier decides a specialist is warranted
    offer_consultation: bool

    # ── Booking / Slot form ─────────────────────────────────────────────────
    patient_name: Optional[str]
    specialty: Optional[str]
    preferred_date: Optional[str]          # ISO date string "YYYY-MM-DD"
    # Next question the slot-collector wants to ask the user
    slot_prompt: Optional[str]
    # Whether all required booking fields have been collected
    slots_complete: bool

    # ── Calendar / Availability ─────────────────────────────────────────────
    # List of {slot_id, doctor, specialty, date, time} dicts
    available_slots: list[dict[str, Any]]

    # ── Human-in-the-Loop ───────────────────────────────────────────────────
    # Slot chosen by the user in the Streamlit UI
    selected_slot: Optional[dict[str, Any]]
    # True after the user clicks "Confirm Booking"
    booking_confirmed: bool

    # ── DB Write ────────────────────────────────────────────────────────────
    booking_record: Optional[dict[str, Any]]

    # ── Misc ────────────────────────────────────────────────────────────────
    error: Optional[str]
    # Routing signal set by nodes to control conditional edges
    next_node: Optional[str]
    # Final message rendered in the chat UI for this turn
    response_text: str
