"""
MedRAG LangGraph Agent
-----------------------
Assembles the full StateGraph for the MedRAG agentic pipeline.

Graph topology
──────────────
START
  └─► intent_router
        ├─► (emergency | off_topic) ──► END
        ├─► clinical ──► rag_node ──► END
        └─► booking  ──► slot_collector
                              ├─► (slots incomplete) ──► [INTERRUPT → await_user] ──► slot_collector
                              └─► (slots complete)   ──► availability_checker
                                                              ├─► (no slots) ──► [INTERRUPT → await_user] ──► slot_collector
                                                              └─► (slots OK) ──► [INTERRUPT → human_in_the_loop]
                                                                                        ├─► (confirmed) ──► db_writer ──► END
                                                                                        └─► (cancelled) ──► END

The graph uses a MemorySaver checkpointer so it can be interrupted
mid-run and resumed with user input from the Streamlit UI.

Usage
─────
    from src.agent.graph import build_graph, run_graph, resume_graph

    graph = build_graph()

    # First invocation
    state = run_graph(graph, user_query="I have chest pain", thread_id="session-123")

    # Resume after user selects a slot in the UI
    state = resume_graph(graph, selected_slot={...}, thread_id="session-123")
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent.intent_router import intent_router_node
from src.agent.rag_node import rag_node
from src.agent.slot_collector import slot_collector_node
from src.agent.calendar_tool import availability_checker_node
from src.agent.db_writer import db_writer_node
from src.utils import get_logger

logger = get_logger(__name__)


# ── Routing helpers ───────────────────────────────────────────────────────────

def _route_after_intent(state: AgentState) -> str:
    intent = state.get("intent", "clinical")
    if intent in ("emergency", "off_topic"):
        return "end_node"
    if intent == "booking":
        return "slot_collector"
    return "rag_node"          # clinical


def _route_after_slot_collector(state: AgentState) -> str:
    if state.get("slots_complete"):
        return "availability_checker"
    return "end_node"          # UI will re-submit next user message → new graph call


def _route_after_availability(state: AgentState) -> str:
    slots = state.get("available_slots", [])
    if not slots:
        return "end_node"      # UI will show "no slots" message, user picks new date
    return "human_in_the_loop_node"


def _route_after_hitl(state: AgentState) -> str:
    if state.get("booking_confirmed"):
        return "db_writer"
    return "end_node"          # User cancelled


# ── Human-in-the-loop passthrough node ───────────────────────────────────────

def human_in_the_loop_node(state: AgentState) -> AgentState:
    """
    Passthrough node. The graph interrupts HERE so the Streamlit UI can
    render the slot-selection card and capture the user's choice.
    The actual confirmation logic lives in resume_graph().
    """
    return {
        **state,
        "next_node": "human_in_the_loop",
        "response_text": "__HITL__",   # Sentinel — UI renders slot card instead
    }


def end_node(state: AgentState) -> AgentState:
    """Terminal passthrough — ensures the graph always ends cleanly."""
    return state


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> Any:
    """Build and compile the MedRAG StateGraph with a MemorySaver checkpointer."""
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("intent_router",          intent_router_node)
    builder.add_node("rag_node",               rag_node)
    builder.add_node("slot_collector",         slot_collector_node)
    builder.add_node("availability_checker",   availability_checker_node)
    builder.add_node("human_in_the_loop_node", human_in_the_loop_node)
    builder.add_node("db_writer",              db_writer_node)
    builder.add_node("end_node",               end_node)

    # Entry point
    builder.set_entry_point("intent_router")

    # Edges
    builder.add_conditional_edges(
        "intent_router",
        _route_after_intent,
        {
            "rag_node":       "rag_node",
            "slot_collector": "slot_collector",
            "end_node":       "end_node",
        },
    )

    builder.add_edge("rag_node", "end_node")

    builder.add_conditional_edges(
        "slot_collector",
        _route_after_slot_collector,
        {
            "availability_checker": "availability_checker",
            "end_node":             "end_node",
        },
    )

    builder.add_conditional_edges(
        "availability_checker",
        _route_after_availability,
        {
            "human_in_the_loop_node": "human_in_the_loop_node",
            "end_node":               "end_node",
        },
    )

    # INTERRUPT at human_in_the_loop_node — graph pauses here for UI input
    builder.add_conditional_edges(
        "human_in_the_loop_node",
        _route_after_hitl,
        {
            "db_writer": "db_writer",
            "end_node":  "end_node",
        },
    )

    builder.add_edge("db_writer", "end_node")
    builder.add_edge("end_node",  END)

    checkpointer = MemorySaver()
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_in_the_loop_node"],
    )
    logger.info("MedRAG agent graph compiled successfully.")
    return graph


# ── Convenience wrappers for the Streamlit UI ─────────────────────────────────

def run_graph(graph, user_query: str, chat_history: list, thread_id: str) -> AgentState:
    """
    Start (or continue) a conversation turn.
    Returns the final state after all non-interrupted nodes have run.
    """
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: AgentState = {
        "user_query":   user_query,
        "chat_history": chat_history,
        # reset per-turn transient fields
        "intent":             None,
        "rag_answer":         "",
        "rag_sources":        [],
        "rag_grounding":      {},
        "offer_consultation": False,
        "slots_complete":     False,
        "slot_prompt":        None,
        "available_slots":    [],
        "selected_slot":      None,
        "booking_confirmed":  False,
        "booking_record":     None,
        "error":              None,
        "next_node":          None,
        "response_text":      "",
    }

    final_state = None
    for event in graph.stream(initial_state, config=config, stream_mode="values"):
        final_state = event

    return final_state or initial_state


def resume_graph(
    graph,
    thread_id: str,
    selected_slot: dict | None = None,
    confirmed: bool = False,
) -> AgentState:
    """
    Resume the graph after the human-in-the-loop interrupt.
    Call with confirmed=True and the selected_slot to proceed to db_writer,
    or confirmed=False to cancel.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Patch state with user's decision before resuming
    update = {
        "selected_slot":     selected_slot,
        "booking_confirmed": confirmed,
    }
    graph.update_state(config, update)

    final_state = None
    for event in graph.stream(None, config=config, stream_mode="values"):
        final_state = event

    return final_state


# ── Singleton graph instance ──────────────────────────────────────────────────

_graph_instance = None


def get_graph():
    """Return the cached singleton graph (built once per Streamlit session)."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_graph()
    return _graph_instance


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the symptoms of hypertension?"
    g = build_graph()
    state = run_graph(g, query, [], thread_id="cli-test")
    print(f"\n🤖 Response:\n{state.get('response_text', '')}")
    print(f"\nIntent: {state.get('intent')}")
    print(f"Offer consultation: {state.get('offer_consultation')}")
    if state.get("available_slots"):
        print(f"Available slots: {len(state['available_slots'])}")
