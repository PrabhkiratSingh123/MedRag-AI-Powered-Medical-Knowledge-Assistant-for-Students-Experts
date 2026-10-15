"""
MedRAG Slot Collector Node
---------------------------
Multi-turn form filler that extracts three required booking fields:
  • patient_name   — full name of the patient
  • specialty      — medical specialty (validated against AVAILABLE_SPECIALTIES)
  • preferred_date — preferred appointment date (ISO "YYYY-MM-DD")

On each invocation the node:
  1. Scans the conversation history + current query for already-provided values.
  2. If all fields are present → sets slots_complete = True.
  3. If fields are missing → generates a natural-language question for the
     next missing field and sets slot_prompt.

Uses Groq to do the extraction so it handles natural language like
"next Tuesday" or "Dr. Smith for my heart" gracefully.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import GROQ_API_KEY, GROQ_MODEL, AVAILABLE_SPECIALTIES
from src.agent.state import AgentState
from src.utils import get_logger

logger = get_logger(__name__)

# ── Extraction prompt ─────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = f"""\
You are a medical appointment assistant. Extract booking details from the conversation.

Today's date is {{today}}.
Available specialties: {', '.join(AVAILABLE_SPECIALTIES)}

From the conversation, extract any of these fields if mentioned:
  patient_name  — full name of the patient (string or null)
  specialty     — must be one of the available specialties (string or null)
  preferred_date — preferred date in ISO format YYYY-MM-DD (string or null)
                  Convert relative dates like "tomorrow", "next Monday" to ISO.

Return ONLY valid JSON with these three keys, no explanation:
{{"patient_name": ..., "specialty": ..., "preferred_date": ...}}
"""

# ── Question prompts for each missing field ───────────────────────────────────

_FIELD_QUESTIONS = {
    "patient_name": "What is the patient's full name?",
    "specialty": (
        f"Which specialist would you like to see?\n"
        f"Available: {', '.join(AVAILABLE_SPECIALTIES)}"
    ),
    "preferred_date": "What date would you prefer for the appointment? (e.g. 15 Oct, next Monday)",
}

_REQUIRED_FIELDS = ["patient_name", "specialty", "preferred_date"]


def _extract_slots(history: list[dict], query: str) -> dict:
    """Call Groq to extract slot values from conversation context."""
    today_str = date.today().isoformat()
    system = _EXTRACT_SYSTEM.replace("{today}", today_str)

    # Build a compact conversation string
    convo_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history[-10:]
    )
    convo_text += f"\nUSER: {query}"

    try:
        llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0, max_tokens=150)
        result = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=convo_text),
        ])
        raw = result.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Slot extraction failed ({e}), returning empty slots")
        return {"patient_name": None, "specialty": None, "preferred_date": None}


def _validate_specialty(specialty: str | None) -> str | None:
    """Fuzzy-match specialty against the allowed list."""
    if not specialty:
        return None
    sl = specialty.lower()
    for s in AVAILABLE_SPECIALTIES:
        if sl in s.lower() or s.lower() in sl:
            return s
    return None


def _validate_date(date_str: str | None) -> str | None:
    """Accept ISO dates; reject past dates."""
    if not date_str:
        return None
    try:
        d = date.fromisoformat(date_str)
        if d < date.today():
            return None          # Past dates not allowed
        return d.isoformat()
    except ValueError:
        return None


def slot_collector_node(state: AgentState) -> AgentState:
    """
    LangGraph node: collect patient_name, specialty, preferred_date.
    Sets slots_complete=True when all fields are gathered,
    or sets slot_prompt with the next question to ask.
    """
    query = state.get("user_query", "")
    history = state.get("chat_history", [])

    # Carry forward any already-collected values from previous turns
    existing_name = state.get("patient_name")
    existing_specialty = state.get("specialty")
    existing_date = state.get("preferred_date")

    logger.info("Slot collector — extracting booking fields from conversation")

    # ── Run Groq extraction ───────────────────────────────────────────────────
    extracted = _extract_slots(history, query)

    patient_name  = existing_name      or extracted.get("patient_name")
    specialty     = _validate_specialty(existing_specialty or extracted.get("specialty"))
    preferred_date = _validate_date(existing_date or extracted.get("preferred_date"))

    logger.info(f"Slots: name={patient_name}, specialty={specialty}, date={preferred_date}")

    # ── Check completeness ────────────────────────────────────────────────────
    missing = []
    if not patient_name:
        missing.append("patient_name")
    if not specialty:
        missing.append("specialty")
    if not preferred_date:
        missing.append("preferred_date")

    if not missing:
        # All slots filled — proceed to availability check
        return {
            **state,
            "patient_name": patient_name,
            "specialty": specialty,
            "preferred_date": preferred_date,
            "slots_complete": True,
            "slot_prompt": None,
            "next_node": "availability_checker",
        }

    # Still missing fields — ask for the next one
    next_field = missing[0]
    prompt = _FIELD_QUESTIONS[next_field]

    return {
        **state,
        "patient_name": patient_name,
        "specialty": specialty,
        "preferred_date": preferred_date,
        "slots_complete": False,
        "slot_prompt": prompt,
        "response_text": prompt,
        "next_node": "await_user",     # Graph pauses here and waits for next user message
    }
