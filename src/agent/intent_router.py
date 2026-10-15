"""
MedRAG Intent Router Node
--------------------------
Classifies each user query into one of four intents:
  • "clinical"   — question about symptoms, conditions, drugs, treatments
  • "booking"    — wants to find / book / consult a doctor
  • "emergency"  — acute danger signal (re-uses existing guardrail)
  • "off_topic"  — clearly outside the medical domain

Uses the existing regex guardrails for emergency / off-topic detection, then
falls back to a lightweight Groq LLM call for the clinical vs. booking split.
"""

from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import GROQ_API_KEY, GROQ_MODEL
from src.guardrails import _EMERGENCY_REGEX, _OFF_TOPIC_REGEX
from src.agent.state import AgentState
from src.utils import get_logger

logger = get_logger(__name__)

_ROUTER_SYSTEM = """\
You are an intent classifier for a medical AI assistant.
Given the user's message, classify it into EXACTLY ONE of these intents:
  clinical  — asks about symptoms, diagnoses, medications, conditions, test results, treatments
  booking   — wants to find a doctor, book an appointment, consult a specialist, or get a referral

Respond with a single lowercase word: clinical  OR  booking
No explanation, no punctuation.
"""


def _llm_classify(query: str) -> str:
    """Use Groq to classify clinical vs. booking intent."""
    try:
        # Always read fresh from .env so model changes take effect immediately
        import os as _os
        from dotenv import load_dotenv as _load_dotenv
        from pathlib import Path as _Path
        _env = _Path(__file__).resolve().parent.parent.parent / ".env"
        for _k in ("GROQ_MODEL", "GROQ_API_KEY"):
            _os.environ.pop(_k, None)
        _load_dotenv(dotenv_path=_env, override=True)
        _model = _os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        _key   = _os.getenv("GROQ_API_KEY", GROQ_API_KEY)

        llm = ChatGroq(api_key=_key, model=_model, temperature=0, max_tokens=5)
        result = llm.invoke([
            SystemMessage(content=_ROUTER_SYSTEM),
            HumanMessage(content=query),
        ])
        label = result.content.strip().lower()
        if label in ("clinical", "booking"):
            return label
        # Fallback: look for booking keywords
        booking_kws = {"book", "appointment", "doctor", "consult", "specialist",
                       "schedule", "available", "slot", "refer", "referral"}
        if any(kw in query.lower() for kw in booking_kws):
            return "booking"
        return "clinical"
    except Exception as e:
        logger.warning(f"Intent router LLM call failed ({e}), defaulting to 'clinical'")
        return "clinical"



def intent_router_node(state: AgentState) -> AgentState:
    """
    LangGraph node: classify intent and set state["intent"].
    Also short-circuits for emergency / off-topic with a ready response.
    """
    query = state.get("user_query", "")
    logger.info(f"Intent router — query: '{query[:80]}'")

    # 1. Emergency (highest priority — regex-based, fast)
    if _EMERGENCY_REGEX.search(query):
        logger.warning("🚨 Emergency intent detected")
        return {
            **state,
            "intent": "emergency",
            "next_node": "end",
            "response_text": (
                "🚨 **This appears to be a medical emergency.**\n\n"
                "**Please call emergency services immediately (911, 999, or your local number).**\n\n"
                "Do not wait for an AI response — go to your nearest emergency room or "
                "call for an ambulance right now.\n\n"
                "---\n*MedRAG cannot assist with medical emergencies.*"
            ),
        }

    # 2. Off-topic (regex-based)
    if _OFF_TOPIC_REGEX.search(query):
        logger.info("Off-topic intent detected")
        return {
            **state,
            "intent": "off_topic",
            "next_node": "end",
            "response_text": (
                "I'm MedRAG, a medical information assistant. I can only help with "
                "health-related questions or booking a doctor consultation.\n\n"
                "Your question appears to be outside my domain."
            ),
        }

    # 3. Clinical vs. Booking — LLM call
    intent = _llm_classify(query)
    logger.info(f"Intent classified as: '{intent}'")
    return {
        **state,
        "intent": intent,
        "next_node": intent,   # routes the graph edge
        "offer_consultation": False,
    }
