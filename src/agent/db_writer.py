"""
MedRAG DB Writer Node
----------------------
Writes a confirmed appointment to MongoDB `appointments` collection.
Called after the user approves a slot in the Streamlit human-in-the-loop card.
"""

from __future__ import annotations

from datetime import datetime

from src.config import APPOINTMENTS_COLLECTION, MONGODB_DB_NAME, get_mongo_client
from src.agent.state import AgentState
from src.utils import get_logger

logger = get_logger(__name__)


def db_writer_node(state: AgentState) -> AgentState:
    """
    LangGraph node: persist the confirmed booking to MongoDB and
    build a rich confirmation message.
    """
    slot = state.get("selected_slot")
    if not slot:
        return {
            **state,
            "error": "No slot selected for booking.",
            "response_text": "⚠️ No slot was selected. Please try again.",
            "next_node": "end",
        }

    record = {
        "slot_id":       slot["slot_id"],
        "patient_name":  state.get("patient_name"),
        "specialty":     slot["specialty"],
        "doctor":        slot["doctor"],
        "date":          slot["date"],
        "time":          slot["time"],
        "booked_at":     datetime.utcnow().isoformat(),
        "status":        "confirmed",
    }

    # ── Write to MongoDB ───────────────────────────────────────────────────
    try:
        client = get_mongo_client()
        try:
            collection = client[MONGODB_DB_NAME][APPOINTMENTS_COLLECTION]
            result = collection.insert_one(record)
            record["_id"] = str(result.inserted_id)
            logger.info(f"Appointment saved: {record['_id']}")
        finally:
            client.close()
    except Exception as e:
        logger.error(f"DB write failed: {e}")
        return {
            **state,
            "error": str(e),
            "response_text": f"⚠️ Failed to save your appointment: {e}",
            "next_node": "end",
        }

    # ── Build confirmation message ─────────────────────────────────────────
    confirmation = (
        f"✅ **Appointment Confirmed!**\n\n"
        f"| Field | Details |\n"
        f"|---|---|\n"
        f"| **Patient** | {record['patient_name']} |\n"
        f"| **Doctor** | {record['doctor']} |\n"
        f"| **Specialty** | {record['specialty']} |\n"
        f"| **Date** | {record['date']} |\n"
        f"| **Time** | {record['time']} |\n\n"
        f"Your booking ID is `{record['_id']}`. "
        f"Please arrive 10 minutes early. 🏥"
    )

    return {
        **state,
        "booking_confirmed": True,
        "booking_record": record,
        "response_text": confirmation,
        "next_node": "end",
    }
