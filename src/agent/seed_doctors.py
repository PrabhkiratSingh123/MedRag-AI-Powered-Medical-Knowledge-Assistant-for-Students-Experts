"""
MedRAG Doctor Roster Seeder
----------------------------
One-shot script that populates the MongoDB `doctors` collection with a
demo roster used by the mock calendar.

Run once at project setup:
    python -m src.agent.seed_doctors

Existing documents are replaced (upsert by doctor name + specialty).
"""

from __future__ import annotations

from src.config import DOCTORS_COLLECTION, MONGODB_DB_NAME, get_mongo_client
from src.utils import get_logger

logger = get_logger(__name__)

DOCTOR_ROSTER = [
    # General Physicians
    {"name": "Dr. Priya Sharma",    "specialty": "General Physician",  "available_days": ["Monday", "Wednesday", "Friday"]},
    {"name": "Dr. Rohan Mehta",     "specialty": "General Physician",  "available_days": ["Tuesday", "Thursday", "Saturday"]},
    # Cardiologists
    {"name": "Dr. Aisha Khan",      "specialty": "Cardiologist",       "available_days": ["Monday", "Tuesday", "Thursday"]},
    {"name": "Dr. Samuel Osei",     "specialty": "Cardiologist",       "available_days": ["Wednesday", "Friday"]},
    # Neurologists
    {"name": "Dr. Li Wei",          "specialty": "Neurologist",        "available_days": ["Monday", "Wednesday", "Friday"]},
    {"name": "Dr. Fatima Al-Zahra", "specialty": "Neurologist",        "available_days": ["Tuesday", "Thursday"]},
    # Diabetologists
    {"name": "Dr. Karthik Nair",    "specialty": "Diabetologist",      "available_days": ["Tuesday", "Thursday", "Saturday"]},
    {"name": "Dr. Maya Patel",      "specialty": "Diabetologist",      "available_days": ["Monday", "Wednesday"]},
    # Pulmonologists
    {"name": "Dr. Carlos Rivera",   "specialty": "Pulmonologist",      "available_days": ["Monday", "Tuesday", "Friday"]},
    {"name": "Dr. Nisha Verma",     "specialty": "Pulmonologist",      "available_days": ["Wednesday", "Thursday"]},
    # Orthopedists
    {"name": "Dr. James Okonkwo",   "specialty": "Orthopedist",        "available_days": ["Monday", "Wednesday", "Friday"]},
    {"name": "Dr. Sneha Gupta",     "specialty": "Orthopedist",        "available_days": ["Tuesday", "Thursday"]},
    # Dermatologists
    {"name": "Dr. Zara Ahmed",      "specialty": "Dermatologist",      "available_days": ["Tuesday", "Thursday", "Saturday"]},
    {"name": "Dr. Ben Nakamura",    "specialty": "Dermatologist",      "available_days": ["Monday", "Wednesday"]},
    # Psychiatrists
    {"name": "Dr. Elena Vasquez",   "specialty": "Psychiatrist",       "available_days": ["Monday", "Wednesday", "Friday"]},
    {"name": "Dr. Arjun Singh",     "specialty": "Psychiatrist",       "available_days": ["Tuesday", "Thursday"]},
]


def seed_doctors() -> int:
    """Upsert the doctor roster into MongoDB. Returns number of upserted docs."""
    client = get_mongo_client()
    try:
        collection = client[MONGODB_DB_NAME][DOCTORS_COLLECTION]
        upserted = 0
        for doc in DOCTOR_ROSTER:
            result = collection.update_one(
                {"name": doc["name"], "specialty": doc["specialty"]},
                {"$set": doc},
                upsert=True,
            )
            if result.upserted_id or result.modified_count:
                upserted += 1
        logger.info(f"✅ Seeded {upserted} doctor(s) into '{DOCTORS_COLLECTION}'")
        return upserted
    finally:
        client.close()


if __name__ == "__main__":
    seed_doctors()
    print("Doctor roster seeded successfully.")
