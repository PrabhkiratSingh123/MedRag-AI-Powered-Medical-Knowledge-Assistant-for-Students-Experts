"""
MedRAG Calendar Tool
---------------------
Mock-first calendar integration.

CALENDAR_MODE = "mock"  (default)
    Queries the MongoDB `doctors` collection to find available slots.
    Each doctor document stores their weekly schedule.
    Slots are generated deterministically for the requested date.

CALENDAR_MODE = "google"  (future)
    Will use Google Calendar API with service-account or OAuth2 credentials.
    Swap `get_available_slots()` implementation when ready.

The slot shape returned is always:
    {
        "slot_id":   str,          # unique booking key
        "doctor":    str,          # "Dr. Aisha Khan"
        "specialty": str,
        "date":      str,          # "YYYY-MM-DD"
        "time":      str,          # "10:30 AM"
    }
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta

from src.config import (
    CALENDAR_MODE,
    DOCTORS_COLLECTION,
    MONGODB_DB_NAME,
    get_mongo_client,
)
from src.agent.state import AgentState
from src.utils import get_logger

logger = get_logger(__name__)


# ── Time-slot generation ──────────────────────────────────────────────────────

_SLOT_TIMES = [
    "09:00 AM", "09:30 AM", "10:00 AM", "10:30 AM",
    "11:00 AM", "11:30 AM", "02:00 PM", "02:30 PM",
    "03:00 PM", "03:30 PM", "04:00 PM", "04:30 PM",
]


def _weekday_name(iso_date: str) -> str:
    """Return day name e.g. 'Monday' for a given ISO date."""
    return date.fromisoformat(iso_date).strftime("%A")


def _mock_get_slots(specialty: str, preferred_date: str) -> list[dict]:
    """
    Query MongoDB doctors collection for doctors of the given specialty
    who are available on the preferred_date weekday, then generate slot times.
    """
    day_name = _weekday_name(preferred_date)

    client = get_mongo_client()
    try:
        collection = client[MONGODB_DB_NAME][DOCTORS_COLLECTION]
        doctors = list(collection.find({
            "specialty": specialty,
            "available_days": day_name,
        }))
    finally:
        client.close()

    if not doctors:
        logger.info(f"No doctors found for {specialty} on {day_name}")
        return []

    slots = []
    for doc in doctors:
        # Give each doctor up to 4 slots for that day
        for time_str in _SLOT_TIMES[:4]:
            slots.append({
                "slot_id": str(uuid.uuid4()),
                "doctor": doc["name"],
                "specialty": specialty,
                "date": preferred_date,
                "time": time_str,
            })

    logger.info(f"Generated {len(slots)} mock slot(s) for {specialty} on {preferred_date}")
    return slots


def get_available_slots(specialty: str, preferred_date: str) -> list[dict]:
    """
    Public interface — dispatches to mock or Google Calendar based on config.
    Returns list of slot dicts.
    """
    if CALENDAR_MODE == "google":
        # Placeholder — integrate Google Calendar API here when ready
        logger.warning("Google Calendar mode not yet implemented; falling back to mock.")

    return _mock_get_slots(specialty, preferred_date)


# ── LangGraph node ────────────────────────────────────────────────────────────

def availability_checker_node(state: AgentState) -> AgentState:
    """
    LangGraph node: fetch available slots for the collected specialty + date.
    Sets state["available_slots"] and routes to human_in_the_loop.
    """
    specialty = state.get("specialty", "")
    preferred_date = state.get("preferred_date", "")

    logger.info(f"Checking availability: {specialty} on {preferred_date}")

    slots = get_available_slots(specialty, preferred_date)

    if not slots:
        # No slots available — ask for a different date
        return {
            **state,
            "available_slots": [],
            "slots_complete": False,
            "preferred_date": None,            # reset so slot_collector re-asks
            "slot_prompt": (
                f"Sorry, no {specialty} slots are available on {preferred_date}. "
                "Please suggest another date."
            ),
            "response_text": (
                f"⚠️ No available {specialty} slots found on **{preferred_date}**. "
                "Please try a different date."
            ),
            "next_node": "await_user",
        }

    return {
        **state,
        "available_slots": slots,
        "next_node": "human_in_the_loop",
    }
